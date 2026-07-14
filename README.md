# trtc

Compile PyTorch models to TensorRT engines: export locally with your
project's torch, build remotely on deployment-class hardware, serve with
manifest-validated engines. The package and its docs live in
[`trtc/`](trtc/README.md).

## Consuming

```toml
# pyproject.toml of your project
[tool.uv.sources]
trtc = { git = "https://github.com/dialohq/trtc", subdirectory = "trtc" }
```

Your `uv.lock` pins the exact trtc commit and your TensorRT version; the
client reads that lock to pick which builder image to use.

## Builder images

Pure C++ (see [`trtc-server/`](trtc-server/README.md) — the spec format, the
tar API, and every `IBuilderConfig` option are documented there) — no Python
in the image, ~2GB. CI publishes one image per supported TensorRT version
(engines are TRT-version-locked), each under three tags:

- `trt10.13` — the moving latest for that TensorRT line
- `1.0.0-trt10.13` — this trtc release
- `1.0.0-trt10.13-<nix hash>` — immutable, content-addressed
- `trt11.1-sm120` (TRT ≥10.16) — single-GPU-architecture, ~700MB instead of
  ~2.5GB; builds for that arch family only (same three tag forms)

The supported version list is the `tensorrtPins` attrset in
[`flake.nix`](flake.nix) — nowhere else. One-shot local builds on a GPU box
need no image at all:

```sh
nix run github:dialohq/trtc#build-10.13 -- spec.json model.onnx [data files...] [--out DIR]
```

Rent a GPU running the version matched to *your* project's lock:

```sh
eval "$(nix run github:dialohq/trtc#launch-builder)"   # run from your project dir
uv run trtc compile <entry> <weights> --builder "$TRTC_BUILDER"
```
