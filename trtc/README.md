# trtc

Compile PyTorch models to TensorRT engines. Export runs locally with your
project's torch; the engine build runs on deployment-class hardware with the
TensorRT version your lock pins; the runtime refuses engines built for a
different TensorRT or GPU arch.

## The three stages

| stage | needs | produces |
|---|---|---|
| **export** | project env (exact locked torch), model code, any CUDA GPU | `*.onnx` + `plan.json` |
| **build** | GPU matching deployment arch, `tensorrt-cu12` matching the pin — no torch, no model code | `*.engine` + `manifest.json` |
| **serve** | `trtc.runtime` validates the manifest (TRT version, compute capability) before loading engines | |

The TensorRT pin is read from **`uv.lock`** (nearest lock wins; `--trt-version`
overrides). `trtc` itself declares no dependencies.

## I have a torch model

Declare a `Bundle` once, next to the model — components, named inputs, dynamic
axes. See `tinfer/.../modules/trt_bundle.py` for a real one (coupled axes,
export-mode rewrites, finalize hook). Then:

```sh
uv run trtc compile <entry.py or module:attr> <weights> --builder http://builder:8080
```

Or split it: `trtc export ... --out ./work`, then `trtc submit ./work ...`.

## I have an ONNX file already

Write a `trtc_build_spec.json` next to it (or don't — a bare ONNX builds with
defaults: strongly typed, TensorRT defaults, no profiles), then point submit
at it:

```sh
uv run trtc submit model.onnx --builder http://builder:8080 --out .
```

## The build spec

All build options live in **`trtc_build_spec.json`**, sitting next to the
ONNX it references — the entire `tensorrt.IBuilderConfig`, as JSON. Nothing
travels as CLI or query parameters:

```json
{
  "trtc_build_spec": 1,
  "components": [
    {
      "onnx": "model.onnx",
      "strongly_typed": true,
      "profiles": [
        {"input": {"min": [1, 80], "opt": [8, 80], "max": [16, 80]}},
        {"input": {"min": [32, 80], "opt": [64, 80], "max": [64, 80]}}
      ],
      "builder_config": {
        "flags": ["TF32", "SPARSE_WEIGHTS"],
        "memory_pool_limits": {"WORKSPACE": "8G"},
        "builder_optimization_level": 5,
        "profiling_verbosity": "DETAILED",
        "hardware_compatibility_level": "AMPERE_PLUS"
      },
      "onnx_sha256": "…optional, verified…",
      "external_data": {"model.onnx.data": null}
    }
  ]
}
```

Each entry of `profiles` is one whole optimization profile covering every
dynamic input; list several to build a multi-profile engine. `builder_config`
covers every `IBuilderConfig` option expressible as data — flags, memory
pools, optimization level, DLA, tactic sources, preview features, tiling,
hardware compatibility; unknown names fail the job loudly with the list the
builder's TensorRT actually has. `external_data` lists the external weight
files of >2GB models (a sibling `model.onnx.data` is picked up by convention);
they travel with the ONNX in every job. `trtc export` writes the spec from
your `Bundle` declaration (`Component.builder_config` passes through); you
can also just write the JSON by hand.

## The builder

`ghcr.io/dialohq/trtc-builder` — **pure C++, no Python in the image**: the
HTTP broker (`trtc-server`) and the build tool (`trtc-build`), linked against
one TensorRT from NVIDIA's official tarball, hash-pinned in `flake.nix` (the
`tensorrtPins` attrset is the single list of supported versions). See
[`trtc-server/`](../trtc-server/README.md). Like a nix derivation, the image
is pinned to one TensorRT version; a plan pinning a different version fails
the job loudly (you run a builder image built for that version instead).
Building locally on a GPU box without the image is
`nix run .#trtc-server-trt10-13 -- ...` territory — the same `trtc-build`
binary the image ships.

### Launch one on vast.ai

`nix run .#launch-builder` rents a GPU, starts the builder image on it, waits
until it answers, and prints the URL to point `trtc` at:

```sh
eval "$(nix run .#launch-builder -- --gpu RTX_4090 --token "$MY_TOKEN")"
# -> sets TRTC_BUILDER=http://<ip>:<port>
uv run trtc compile <entry> <weights> --builder "$TRTC_BUILDER" --token "$MY_TOKEN"
```

It needs a vast.ai key (`VAST_API_KEY` or a configured `vastai`); options:
`--image`, `--gpu`, `--disk`, `--idle-timeout` (self-shutdown), `--login`
(for a private registry), `--query` (full vast offer query). It prints the
`vastai destroy instance <id>` command to tear it down.

### The API

Deliberately dumb: one job is one tar — `trtc_build_spec.json` with its ONNX
(and any external weight data files) next to it, the exact on-disk layout —
returning one engine. No build parameters in the request. Multi-component
models are composed client-side. Engine + timing caches persist under
`TRTC_DATA_DIR`, so any HTTP client works:

```sh
tar cf job.tar trtc_build_spec.json model.onnx model.onnx.data
curl -X POST -H 'Content-Type: application/x-tar' \
    --data-binary @job.tar "http://builder:8080/builds"
```

`GET /info` reports the TensorRT the image was baked with; the client checks
it against your `uv.lock` pin before submitting.

```sh
# vast.ai
vastai create instance <offer> --image ghcr.io/dialohq/trtc-builder:trt11.1 \
    --disk 40 --env '-p 8080:8080 -e TRTC_TOKEN=...'

# anywhere with a GPU
docker run --gpus all -p 8080:8080 -v trtc-data:/data ghcr.io/dialohq/trtc-builder
```

Set `TRTC_TOKEN` for auth, `TRTC_IDLE_TIMEOUT` for self-shutdown on idle.
`trtc inspect <dir>` pretty-prints plans/manifests; `trtc info` shows local
GPU + TRT facts.
