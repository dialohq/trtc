# trtc

Compile PyTorch models to TensorRT engines: export locally with your
project's torch, build remotely on deployment-class hardware, serve with
manifest-validated engines. A uv workspace of two packages: the client
[`trtc/`](trtc/README.md) (docs live there) and the builder
[`trtc-server/`](trtc-server/README.md) — the builder image is exactly the
`trtc-server` member installed with its locked dependencies.

## Consuming

```toml
# pyproject.toml of your project
[tool.uv.sources]
trtc = { git = "https://github.com/dialohq/trtc", subdirectory = "trtc" }
```

Your `uv.lock` pins the exact trtc commit and your TensorRT version; trtc
reads that lock for every version decision.

## Builder images

CI publishes one image per supported TensorRT version (engines are
TRT-version-locked): `ghcr.io/dialohq/trtc-builder:trt<major.minor>`. The
supported version list is the matrix in
[`.github/workflows/ci.yml`](.github/workflows/ci.yml).

Rent a GPU running the version matched to *your* project's lock:

```sh
eval "$(nix run github:dialohq/trtc#launch-builder)"   # run from your project dir
uv run trtc compile <entry> <weights> --builder "$TRTC_BUILDER"
```
