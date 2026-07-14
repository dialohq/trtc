"""`trtc-server` — the builder CLI: build engines, run the build server.

Runs in an environment with the pinned TensorRT installed (the builder image,
or any GPU box). Never imports torch or model code. All build options live in
trtc_build_spec.json next to the ONNX (see trtc.buildspec) — the CLI only
says where things are.
"""

from __future__ import annotations

import argparse

from ..buildspec import MANIFEST_FILE, resolve_build_target


def cmd_build(args: argparse.Namespace) -> None:
    from .build import build_spec

    try:
        work_dir, spec = resolve_build_target(args.target)
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error
    build_spec(spec, work_dir, args.out, force=args.force, timing_cache_path=args.timing_cache)
    print(f"engines + {MANIFEST_FILE} written to {args.out or work_dir}")


def cmd_serve(args: argparse.Namespace) -> None:
    from .app import serve

    serve(host=args.host, port=args.port)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="trtc-server", description="Build TensorRT engines from trtc build specs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build", help="Spec dir or bare ONNX -> engines (needs GPU + pinned TRT)"
    )
    build_parser.add_argument(
        "target",
        help="Directory containing trtc_build_spec.json, or a bare .onnx "
        "(uses the spec next to it, or defaults if there is none)",
    )
    build_parser.add_argument("--out", default=None, help="Engine output directory (default: alongside input)")
    build_parser.add_argument("--force", action="store_true", help="Rebuild even if engines exist")
    build_parser.add_argument("--timing-cache", default=None, help="TensorRT timing cache file")
    build_parser.set_defaults(handler=cmd_build)

    serve_parser = subparsers.add_parser("serve", help="Run the builder server (see trtc/server/app.py for env vars)")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8080)
    serve_parser.set_defaults(handler=cmd_serve)

    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
