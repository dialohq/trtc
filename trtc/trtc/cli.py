"""trtc command line: export, build, compile (export+build), submit, serve, inspect."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .plan import (
    MANIFEST_FILE,
    plan_for_onnx,
    query_gpu,
    read_json,
    read_plan,
    resolve_tensorrt_version,
    write_json,
)
from .spec import Bundle, load_entry, parse_options


def parse_shape_profile(spec: str) -> tuple[str, dict[str, list[int]]]:
    """'asr=1x512x128:8x512x256:16x512x1024' -> ('asr', {min/opt/max: [...]})"""
    name, sep, ranges = spec.partition("=")
    parts = ranges.split(":") if sep else []
    if not name or len(parts) != 3:
        raise SystemExit(f"--shape expects NAME=MIN:OPT:MAX with 'x'-separated dims, got {spec!r}")
    shapes = [[int(dim) for dim in part.split("x")] for part in parts]
    if len({len(shape) for shape in shapes}) != 1:
        raise SystemExit(f"--shape {name}: min/opt/max must have the same rank, got {spec!r}")
    return name, {"min": shapes[0], "opt": shapes[1], "max": shapes[2]}


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
        tensorrt_version=resolve_tensorrt_version(args.trt_version),
        provenance={"entry": args.entry, "weights": list(args.weights), "options": options},
    )
    print(f"plan + ONNX written to {out_dir}")
    return out_dir


def _finalize(bundle: Bundle, engine_dir: Path) -> None:
    if bundle.finalize is None:
        return
    manifest = read_json(engine_dir / MANIFEST_FILE)
    bundle.finalize(manifest, engine_dir)


def cmd_export(args: argparse.Namespace) -> None:
    bundle, options = _load_bundle(args)
    _run_export(bundle, options, args)


def _resolve_build_target(args: argparse.Namespace) -> tuple[Path, dict]:
    """A build/submit target is a plan directory (from `trtc export`), or a
    bare .onnx for which a single-component plan is synthesized in memory."""
    target = Path(args.target)
    if target.suffix != ".onnx":
        if args.shape or args.name:
            raise SystemExit("--shape/--name only apply when the target is a bare .onnx file")
        return target, read_plan(target)
    if not target.exists():
        raise SystemExit(f"ONNX file not found: {target}")
    onnx_path = target.resolve()
    plan = plan_for_onnx(
        onnx_path,
        tensorrt_version=resolve_tensorrt_version(args.trt_version),
        name=args.name,
        dtype=args.dtype,
        workspace_gb=args.workspace_gb,
        profiles=dict(parse_shape_profile(spec) for spec in (args.shape or [])),
    )
    return onnx_path.parent, plan


def _submit_plan(
    builder: str,
    plan: dict,
    work_dir: Path,
    out_dir: Path,
    *,
    token: str | None,
    output_url: str | None = None,
) -> dict | None:
    """One builder job per component; the builder itself never sees more than
    one ONNX at a time. With output_url the builder PUTs the engine there
    (single-component only); otherwise engines and the assembled manifest are
    downloaded into out_dir."""
    from .plan import assemble_manifest, build_params
    from .remote import download_engine, submit_build, wait_for_build

    components = plan["components"]
    if output_url is not None and len(components) != 1:
        raise SystemExit(
            f"--output-url takes a single engine but this target has {len(components)} components; "
            "omit it to download engines locally."
        )

    results = []
    for component in components:
        job_id = submit_build(
            builder,
            work_dir / component["onnx"],
            params=build_params(component, plan["tensorrt_version"]),
            output_url=output_url,
            token=token,
        )
        print(f"submitted {component['name']} as job {job_id}")
        job = wait_for_build(builder, job_id, token=token)
        if job["state"] != "succeeded":
            raise SystemExit(f"remote build of {component['name']} failed: {job.get('error', 'see builder log')}")
        if output_url is not None:
            continue  # builder uploaded the engine; nothing to download or assemble
        if not job.get("result"):
            raise SystemExit(f"builder returned no build result for {component['name']}")
        download_engine(builder, job_id, out_dir / component["engine"], token=token)
        results.append(job["result"])

    if output_url is not None:
        return None
    manifest = assemble_manifest(plan, results)
    write_json(out_dir / MANIFEST_FILE, manifest)
    return manifest


def _add_onnx_target_arguments(parser: argparse.ArgumentParser, *, target_optional: bool = False) -> None:
    target_kwargs = {"nargs": "?", "default": None} if target_optional else {}
    parser.add_argument("target", help="Plan directory (from `trtc export`) or a bare .onnx file", **target_kwargs)
    parser.add_argument(
        "--shape",
        action="append",
        metavar="NAME=MIN:OPT:MAX",
        help="Bare-ONNX: optimization profile per dynamic input, e.g. x=1x80:8x80:16x80 (repeatable)",
    )
    parser.add_argument("--name", default=None, help="Bare-ONNX: component name (default: file stem)")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float32", help="Bare-ONNX only")
    parser.add_argument("--workspace-gb", type=float, default=4.0, help="Bare-ONNX only")
    parser.add_argument("--trt-version", default=None, help="TensorRT pin (default: uv.lock, then installed)")


def cmd_build(args: argparse.Namespace) -> None:
    from .build import build_plan

    work_dir, plan = _resolve_build_target(args)
    build_plan(plan, work_dir, args.out, force=args.force, timing_cache_path=args.timing_cache)
    print(f"engines + {MANIFEST_FILE} written to {args.out or work_dir}")


def cmd_compile(args: argparse.Namespace) -> None:
    bundle, options = _load_bundle(args)
    out_dir = _run_export(bundle, options, args)

    if args.builder:
        plan = read_plan(out_dir)
        _submit_plan(args.builder, plan, out_dir, out_dir, token=args.token)
    else:
        from .build import build_plan

        build_plan(read_plan(out_dir), out_dir, force=args.force, timing_cache_path=args.timing_cache)

    _finalize(bundle, out_dir)
    print(f"compiled {bundle.name}: {out_dir / MANIFEST_FILE}")


def cmd_submit(args: argparse.Namespace) -> None:
    if args.target is not None:
        work_dir, plan = _resolve_build_target(args)
        if args.output_url:
            _submit_plan(args.builder, plan, work_dir, work_dir, token=args.token, output_url=args.output_url)
            print(f"engine uploaded to {args.output_url}")
        else:
            out_dir = Path(args.out) if args.out else work_dir
            _submit_plan(args.builder, plan, work_dir, out_dir, token=args.token)
            print(f"engines + {MANIFEST_FILE} written to {out_dir}")
        return

    # Presigned input: the builder pulls one ONNX from input_url; with
    # output_url it PUTs the engine there, otherwise --out downloads it.
    from .remote import download_engine, submit_build, wait_for_build

    if not args.input_url:
        raise SystemExit("Pass a target (plan dir or .onnx) or --input-url")
    if not args.output_url and not args.out:
        raise SystemExit("Presigned input needs --output-url (builder uploads) or --out (client downloads)")
    params = {
        "trt_version": resolve_tensorrt_version(args.trt_version),
        "name": args.name or "model",
        "dtype": args.dtype,
        "workspace_gb": args.workspace_gb,
        "shapes": args.shape or [],
    }
    job_id = submit_build(
        args.builder, params=params, input_url=args.input_url, output_url=args.output_url, token=args.token
    )
    print(f"submitted build {job_id}")
    job = wait_for_build(args.builder, job_id, token=args.token)
    if job["state"] != "succeeded":
        raise SystemExit(f"remote build failed: {job.get('error', 'see builder log')}")
    if not args.output_url:
        dest = Path(args.out) / f"{params['name']}.engine"
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


def cmd_serve(args: argparse.Namespace) -> None:
    from .server import serve

    serve(host=args.host, port=args.port)


def cmd_inspect(args: argparse.Namespace) -> None:
    path = Path(args.path)
    if path.is_dir():
        for candidate in (path / MANIFEST_FILE, path / "plan.json"):
            if candidate.exists():
                path = candidate
                break
        else:
            raise SystemExit(f"No plan.json or {MANIFEST_FILE} in {path}")
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
    parser.add_argument("--trt-version", default=None, help="TensorRT pin (default: installed tensorrt-cu12)")


def _add_builder_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--builder", default=None, help="Builder URL; omit to build locally")
    parser.add_argument("--token", default=None, help="Builder auth token")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="trtc", description="Compile PyTorch models to TensorRT engines.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Model -> ONNX + plan.json (needs the project env)")
    _add_entry_arguments(export_parser)
    export_parser.set_defaults(handler=cmd_export)

    build_parser = subparsers.add_parser("build", help="Plan dir or bare ONNX -> engines (needs GPU + pinned TRT)")
    _add_onnx_target_arguments(build_parser)
    build_parser.add_argument("--out", default=None, help="Engine output directory (default: alongside input)")
    build_parser.add_argument("--force", action="store_true", help="Rebuild even if engines exist")
    build_parser.add_argument("--timing-cache", default=None, help="TensorRT timing cache file")
    build_parser.set_defaults(handler=cmd_build)

    compile_parser = subparsers.add_parser("compile", help="export + build (+ finalize), locally or via --builder")
    _add_entry_arguments(compile_parser)
    _add_builder_arguments(compile_parser)
    compile_parser.add_argument("--force", action="store_true", help="Rebuild even if engines exist")
    compile_parser.add_argument("--timing-cache", default=None, help="TensorRT timing cache file (local builds)")
    compile_parser.set_defaults(handler=cmd_compile)

    submit_parser = subparsers.add_parser("submit", help="Send a plan dir or bare ONNX to a builder")
    _add_onnx_target_arguments(submit_parser, target_optional=True)  # presigned-URL mode has no local target
    submit_parser.add_argument("--builder", required=True, help="Builder URL")
    submit_parser.add_argument("--token", default=None, help="Builder auth token")
    submit_parser.add_argument("--input-url", default=None, help="Presigned GET URL of a plan tarball (instead of upload)")
    submit_parser.add_argument("--output-url", default=None, help="Presigned PUT URL the builder uploads engines to")
    submit_parser.add_argument("--out", default=None, help="Download engines here when no --output-url is set")
    submit_parser.set_defaults(handler=cmd_submit)

    serve_parser = subparsers.add_parser("serve", help="Run the builder server (see trtc/server.py for env vars)")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8080)
    serve_parser.set_defaults(handler=cmd_serve)

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

    inspect_parser = subparsers.add_parser("inspect", help="Pretty-print a plan.json / manifest.json / engine dir")
    inspect_parser.add_argument("path")
    inspect_parser.set_defaults(handler=cmd_inspect)

    info_parser = subparsers.add_parser("info", help="Show local GPU and TensorRT facts")
    info_parser.set_defaults(handler=cmd_info)

    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
