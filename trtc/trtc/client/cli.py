"""`trtc` — the client CLI: export models, submit builds, provision builders.

Engines are always built by a builder (`trtc-server`); this CLI never imports
tensorrt. All build options live in trtc_build_spec.json next to the ONNX
(see trtc.buildspec) — the CLI only says where things are and which builder
to use. To build on the local machine, run `trtc-server build` in an
environment with the pinned TensorRT installed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..buildspec import (
    BUILD_SPEC_FILE,
    MANIFEST_FILE,
    BuildSpec,
    assemble_manifest,
    pack_job_tar,
    query_gpu,
    read_build_spec,
    read_json,
    resolve_build_target,
    resolve_tensorrt_version,
    single_component_spec,
    trt_pin_satisfied,
    write_json,
)
from ..spec import Bundle, load_entry, parse_options


def _load_bundle(args: argparse.Namespace) -> tuple[Bundle, dict[str, Any]]:
    factory = load_entry(args.entry)
    options = parse_options(args.set or [])
    bundle = factory(*args.weights, **options)
    if not isinstance(bundle, Bundle):
        raise SystemExit(f"Entry {args.entry!r} returned {type(bundle).__name__}, expected trtc.Bundle")
    return bundle, options


def _resolve_out_dir(explicit: str | None, bundle: Bundle) -> Path:
    if explicit:
        return Path(explicit)
    if bundle.engine_dir_hint:
        return Path(bundle.engine_dir_hint)
    raise SystemExit("No output directory: pass --out or set Bundle.engine_dir_hint in the entry")


def _run_export(bundle: Bundle, options: dict[str, Any], args: argparse.Namespace) -> Path:
    from .export import export_bundle

    out_dir = _resolve_out_dir(args.out, bundle)
    export_bundle(
        bundle,
        out_dir,
        device=args.device,
        provenance={"entry": args.entry, "weights": list(args.weights), "options": options},
    )
    print(f"{BUILD_SPEC_FILE} + ONNX written to {out_dir}")
    return out_dir


def cmd_export(args: argparse.Namespace) -> None:
    bundle, options = _load_bundle(args)
    _run_export(bundle, options, args)


def _check_builder_trt(builder: str, *, token: str | None) -> None:
    """The builder's TensorRT is prebaked, not pickable — so before wasting a
    build, compare it against this project's uv.lock pin (if there is one)
    and refuse a builder baked with something else."""
    from .remote import builder_info

    try:
        pinned = resolve_tensorrt_version(None)
    except SystemExit:
        return  # no pin anywhere; whatever the builder has is what you get
    prebaked = builder_info(builder, token=token).get("tensorrt")
    if prebaked and not trt_pin_satisfied(pinned, prebaked):
        raise SystemExit(
            f"Builder at {builder} is baked with TensorRT {prebaked} but this project pins {pinned}. "
            "Point at (or `trtc launch`) the builder image built for your pin."
        )


def _submit_spec(
    builder: str,
    spec: BuildSpec,
    work_dir: Path,
    out_dir: Path,
    *,
    token: str | None,
    output_url: str | None = None,
) -> dict | None:
    """One builder job per component; the builder itself never sees more than
    one component at a time. With output_url the builder PUTs the engine
    there (single-component only); otherwise engines and the assembled
    manifest are downloaded into out_dir."""
    from .remote import download_engine, submit_build, wait_for_build

    _check_builder_trt(builder, token=token)

    if output_url is not None and len(spec.components) != 1:
        raise SystemExit(
            f"--output-url takes a single engine but this target has {len(spec.components)} components; "
            "omit it to download engines locally."
        )

    results = []
    for component in spec.components:
        job_tar = pack_job_tar(single_component_spec(spec, component), work_dir)
        job_id = submit_build(builder, job_tar, output_url=output_url, token=token)
        print(f"submitted {component.name} as job {job_id}")
        job = wait_for_build(builder, job_id, token=token)
        if job["state"] != "succeeded":
            raise SystemExit(f"remote build of {component.name} failed: {job.get('error', 'see builder log')}")
        if output_url is not None:
            continue  # builder uploaded the engine; nothing to download or assemble
        if not job.get("result"):
            raise SystemExit(f"builder returned no build result for {component.name}")
        download_engine(builder, job_id, out_dir / component.engine, token=token)
        results.append(job["result"])

    if output_url is not None:
        return None
    manifest = assemble_manifest(spec, results)
    write_json(out_dir / MANIFEST_FILE, manifest)
    return manifest


def cmd_compile(args: argparse.Namespace) -> None:
    bundle, options = _load_bundle(args)
    out_dir = _run_export(bundle, options, args)
    _submit_spec(args.builder, read_build_spec(out_dir), out_dir, out_dir, token=args.token)
    if bundle.finalize is not None:
        bundle.finalize(read_json(out_dir / MANIFEST_FILE), out_dir)
    print(f"compiled {bundle.name}: {out_dir / MANIFEST_FILE}")


def cmd_submit(args: argparse.Namespace) -> None:
    if args.target is not None:
        try:
            work_dir, spec = resolve_build_target(args.target)
        except (ValueError, FileNotFoundError) as error:
            raise SystemExit(str(error)) from error
        if args.output_url:
            _submit_spec(args.builder, spec, work_dir, work_dir, token=args.token, output_url=args.output_url)
            print(f"engine uploaded to {args.output_url}")
        else:
            out_dir = Path(args.out) if args.out else work_dir
            _submit_spec(args.builder, spec, work_dir, out_dir, token=args.token)
            print(f"engines + {MANIFEST_FILE} written to {out_dir}")
        return

    # Presigned input: the builder pulls one job tar from input_url; with
    # output_url it PUTs the engine there, otherwise --out downloads it.
    from .remote import download_engine, submit_build, wait_for_build

    if not args.input_url:
        raise SystemExit(f"Pass a target (spec dir or .onnx) or --input-url (a job tar: {BUILD_SPEC_FILE} + ONNX)")
    if not args.output_url and not args.out:
        raise SystemExit("Presigned input needs --output-url (builder uploads) or --out (client downloads)")
    _check_builder_trt(args.builder, token=args.token)
    job_id = submit_build(args.builder, input_url=args.input_url, output_url=args.output_url, token=args.token)
    print(f"submitted build {job_id}")
    job = wait_for_build(args.builder, job_id, token=args.token)
    if job["state"] != "succeeded":
        raise SystemExit(f"remote build failed: {job.get('error', 'see builder log')}")
    if not args.output_url:
        engine_name = job["spec"]["components"][0]["engine"]
        dest = Path(args.out) / engine_name
        download_engine(args.builder, job_id, dest, token=args.token)
        print(f"engine downloaded to {dest}")


def cmd_launch(args: argparse.Namespace) -> None:
    from . import vast

    trt_version = args.trt_version or resolve_tensorrt_version(None)
    image = args.image or f"{args.registry}:trt{'.'.join(trt_version.split('.')[:2])}"
    url = vast.launch(
        image=image,
        gpu=args.gpu,
        disk=args.disk,
        token=args.token,
        idle_timeout=args.idle_timeout,
        login=args.login,
        offers=args.offers,
        query=args.query,
    )
    # Only the export line goes to stdout, so `eval "$(trtc launch ...)"` works.
    print(f"export TRTC_BUILDER={url}")


def cmd_inspect(args: argparse.Namespace) -> None:
    path = Path(args.path)
    if path.is_dir():
        for candidate in (path / MANIFEST_FILE, path / BUILD_SPEC_FILE):
            if candidate.exists():
                path = candidate
                break
        else:
            raise SystemExit(f"No {BUILD_SPEC_FILE} or {MANIFEST_FILE} in {path}")
    print(json.dumps(read_json(path), indent=2, sort_keys=True))


def cmd_info(args: argparse.Namespace) -> None:
    del args
    info: dict[str, Any] = dict(query_gpu())
    try:
        info["tensorrt_version"] = resolve_tensorrt_version(None)
    except SystemExit:
        info["tensorrt_version"] = None
    print(json.dumps(info, indent=2))


def _add_entry_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("entry", help="Bundle entry: path/to/bundle.py[:attr] or package.module:attr")
    parser.add_argument("weights", nargs="+", help="Weight path(s) passed positionally to the bundle factory")
    parser.add_argument("--set", action="append", metavar="KEY=VALUE", help="Bundle option override (repeatable)")
    parser.add_argument("--out", default=None, help="Output directory (default: bundle's engine_dir_hint)")
    parser.add_argument("--device", default="cuda", help="Device for export tracing (default: cuda)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="trtc", description="Compile PyTorch models to TensorRT engines.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser(
        "export", help=f"Model -> ONNX + {BUILD_SPEC_FILE} (needs the project env)"
    )
    _add_entry_arguments(export_parser)
    export_parser.set_defaults(handler=cmd_export)

    compile_parser = subparsers.add_parser("compile", help="export + build on a builder (+ finalize)")
    _add_entry_arguments(compile_parser)
    compile_parser.add_argument("--builder", required=True, help="Builder URL (see `trtc launch`)")
    compile_parser.add_argument("--token", default=None, help="Builder auth token")
    compile_parser.set_defaults(handler=cmd_compile)

    submit_parser = subparsers.add_parser("submit", help="Send a build spec dir or bare ONNX to a builder")
    submit_parser.add_argument(
        "target",
        nargs="?",
        default=None,  # presigned-URL mode has no local target
        help=f"Directory containing {BUILD_SPEC_FILE}, or a bare .onnx "
        "(uses the spec next to it, or defaults if there is none)",
    )
    submit_parser.add_argument("--builder", required=True, help="Builder URL")
    submit_parser.add_argument("--token", default=None, help="Builder auth token")
    submit_parser.add_argument("--input-url", default=None, help="Presigned GET URL of a job tar (instead of upload)")
    submit_parser.add_argument("--output-url", default=None, help="Presigned PUT URL the builder uploads engines to")
    submit_parser.add_argument("--out", default=None, help="Download engines here when no --output-url is set")
    submit_parser.set_defaults(handler=cmd_submit)

    launch_parser = subparsers.add_parser(
        "launch", help="Rent a vast.ai GPU and start a builder (needs the 'launch' extra: trtc[launch])"
    )
    launch_parser.add_argument("--trt-version", default=None, help="TensorRT version to build for (default: uv.lock)")
    launch_parser.add_argument("--image", default=None, help="Override the builder image entirely")
    launch_parser.add_argument("--registry", default="ghcr.io/dialohq/trtc-builder", help="Builder image registry")
    launch_parser.add_argument("--gpu", default="RTX_4090", help="vast.ai gpu_name filter")
    launch_parser.add_argument("--disk", type=int, default=40, help="Instance disk GB")
    launch_parser.add_argument("--token", default=None, help="Set TRTC_TOKEN on the builder")
    launch_parser.add_argument("--idle-timeout", type=int, default=None, help="Builder self-shutdown after N idle secs")
    launch_parser.add_argument("--login", default=None, help="Registry creds for a private image: '-u USER -p TOKEN host'")
    launch_parser.add_argument("--offers", type=int, default=5, help="How many cheapest offers to try")
    launch_parser.add_argument("--query", default=None, help="Full vast.ai offer query (overrides --gpu/--disk)")
    launch_parser.set_defaults(handler=cmd_launch)

    inspect_parser = subparsers.add_parser(
        "inspect", help=f"Pretty-print a {BUILD_SPEC_FILE} / {MANIFEST_FILE} / engine dir"
    )
    inspect_parser.add_argument("path")
    inspect_parser.set_defaults(handler=cmd_inspect)

    info_parser = subparsers.add_parser("info", help="Show local GPU and TensorRT facts")
    info_parser.set_defaults(handler=cmd_info)

    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
