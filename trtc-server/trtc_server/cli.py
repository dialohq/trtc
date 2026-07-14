"""`trtc-server` — the builder CLI: build engines, run the build server.

Runs in an environment with the pinned TensorRT installed (the builder image,
or any GPU box). Never imports torch or model code.
"""

from __future__ import annotations

import argparse

from trtc.plan import MANIFEST_FILE, resolve_build_target, resolve_tensorrt_version


def cmd_build(args: argparse.Namespace) -> None:
    from .build import build_plan

    try:
        work_dir, plan = resolve_build_target(
            args.target,
            tensorrt_version=resolve_tensorrt_version(args.trt_version),
            name=args.name,
            dtype=args.dtype,
            workspace_gb=args.workspace_gb,
            shapes=args.shape or [],
        )
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error
    build_plan(plan, work_dir, args.out, force=args.force, timing_cache_path=args.timing_cache)
    print(f"engines + {MANIFEST_FILE} written to {args.out or work_dir}")


def cmd_serve(args: argparse.Namespace) -> None:
    from .app import serve

    serve(host=args.host, port=args.port)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="trtc-server", description="Build TensorRT engines from trtc plans.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Plan dir or bare ONNX -> engines (needs GPU + pinned TRT)")
    build_parser.add_argument("target", help="Plan directory (from `trtc export`) or a bare .onnx file")
    build_parser.add_argument(
        "--shape",
        action="append",
        metavar="NAME=MIN:OPT:MAX",
        help="Bare-ONNX: optimization profile per dynamic input, e.g. x=1x80:8x80:16x80 (repeatable)",
    )
    build_parser.add_argument("--name", default=None, help="Bare-ONNX: component name (default: file stem)")
    build_parser.add_argument("--dtype", choices=["float16", "float32"], default="float32", help="Bare-ONNX only")
    build_parser.add_argument("--workspace-gb", type=float, default=4.0, help="Bare-ONNX only")
    build_parser.add_argument("--trt-version", default=None, help="TensorRT pin (default: uv.lock, then installed)")
    build_parser.add_argument("--out", default=None, help="Engine output directory (default: alongside input)")
    build_parser.add_argument("--force", action="store_true", help="Rebuild even if engines exist")
    build_parser.add_argument("--timing-cache", default=None, help="TensorRT timing cache file")
    build_parser.set_defaults(handler=cmd_build)

    serve_parser = subparsers.add_parser("serve", help="Run the builder server (see trtc_server/app.py for env vars)")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8080)
    serve_parser.set_defaults(handler=cmd_serve)

    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
