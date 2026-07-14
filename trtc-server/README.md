# trtc-server (C++)

The builder, in plain C++ with no Python anywhere near the image:

- `trtc-server` — the HTTP job broker (`serve`). cpp-httplib + nlohmann_json
  + OpenSSL; knows nothing about TensorRT. One job = one ONNX + query
  parameters = one engine back, same API the Python client always spoke.
- `trtc-build` — one ONNX -> one engine + manifest.json, linked against the
  one TensorRT the image ships (NVIDIA's official tarball, pinned by hash in
  `flake.nix`). The broker execs it next to itself; a plan pinning a
  different TensorRT fails the job loudly.

Build and test (the broker needs no TensorRT):

```sh
nix build .#trtc-broker
TRTC_SERVER_BIN=$PWD/result/bin/trtc-server \
    uv run --package trtc python -m unittest discover -s trtc-server/tests -v

nix build .#trtc-server-trt10-13     # broker + trtc-build against TRT 10.13
nix build .#trtc-builder-trt10-13    # the image
```

The image is two binaries, one TensorRT layer, CA certs, and nothing else.
Engine builds require the deployment GPU; off-GPU, `trtc-build` fails with a
clear error before touching TensorRT.
