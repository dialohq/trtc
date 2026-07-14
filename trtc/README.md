# trtc

Compile PyTorch models to TensorRT engines. Export runs locally with your
project's torch; the engine build runs on deployment-class hardware with the
TensorRT prebaked into the builder image; the runtime refuses engines built
for a different TensorRT or GPU arch.

## Client and server

The package splits into two sides with two CLIs and no dependency on each
other (they share only the spec/manifest contract in `trtc.buildspec`):

- **`trtc`** (`trtc.client`) — runs where the model code lives: `export`,
  `submit`, `compile` (= export + submit), `launch`, `inspect`, `info`.
  Never imports tensorrt.
- **`trtc-server`** (`trtc.server`) — runs on GPU hardware with a prebaked
  TensorRT: `serve` (the builder HTTP API) and `build` (engines from a spec
  dir or bare ONNX). Never imports torch or model code. The server builds,
  end of story.

## The build spec

All build options live in **`trtc_build_spec.json`**, sitting next to the
ONNX files it references. Nothing travels as CLI or query parameters — the
spec file is the whole vocabulary, and it maps 1:1 onto the TensorRT Python
API (`trtc.BuilderConfig` fields are `tensorrt.IBuilderConfig` attributes,
enum values spelled as member names):

```json
{
  "trtc_build_spec": 1,
  "components": [
    {
      "onnx": "encoder.onnx",
      "strongly_typed": true,
      "profiles": [
        {"tokens": {"min": [1, 128], "opt": [8, 256], "max": [16, 1024]},
         "mask":   {"min": [1, 128], "opt": [8, 256], "max": [16, 1024]}},
        {"tokens": {"min": [32, 128], "opt": [64, 256], "max": [64, 1024]},
         "mask":   {"min": [32, 128], "opt": [64, 256], "max": [64, 1024]}}
      ],
      "builder_config": {
        "flags": ["FP16", "SPARSE_WEIGHTS"],
        "memory_pool_limits": {"WORKSPACE": "8G"},
        "builder_optimization_level": 5,
        "profiling_verbosity": "DETAILED",
        "hardware_compatibility_level": "AMPERE_PLUS"
      }
    }
  ]
}
```

That's the whole schema — a version, and per ONNX: `profiles`,
`builder_config`, `strongly_typed`, plus optional integrity fields
(`onnx_sha256`, `external_data`). Component and engine names derive from the
ONNX file name; there is no other metadata.

Each entry of `profiles` is one whole optimization profile covering every
dynamic input; list several to build a multi-profile engine. `builder_config`
keys are `IBuilderConfig` attributes — anything TensorRT exposes as data
works, without trtc naming it in advance; unknown keys fail at parse time,
and unknown flag/pool/enum names fail the build loudly with the list of names
the builder's TensorRT actually has. Unset means TensorRT's own default.
Options that are live Python objects (`int8_calibrator`, `algorithm_selector`,
`profile_stream`) are not data and cannot appear in a spec.

### Large models (external weight data)

Models over 2GB can't keep their weights inside the ONNX protobuf — they ship
as `model.onnx` plus external data files next to it. A component lists those
files under `external_data` (file name → sha256, or `null` to skip
verification); they travel with the ONNX in every job tar, and the builder
parses the model from disk so TensorRT resolves them. `trtc export` detects
the files torch writes and fills this in automatically; a bare
`model.onnx` with a sibling `model.onnx.data` is picked up by convention.

```json
{"onnx": "big.onnx", "external_data": {"big.onnx.data": null}}
```

The same shapes exist in Python as dataclasses (`trtc.BuildSpec`,
`trtc.ComponentSpec`, `trtc.BuilderConfig`, `trtc.ShapeRange`) — `trtc export`
writes the file from your `Bundle` declaration; you can also just write the
JSON by hand.

## The three stages

| stage | needs | produces |
|---|---|---|
| **export** | project env (exact locked torch), model code, any CUDA GPU | `*.onnx` + `trtc_build_spec.json` |
| **build** | GPU matching deployment arch, a prebaked TensorRT — no torch, no model code | `*.engine` + `manifest.json` |
| **serve** | `trtc.runtime` validates the manifest (TRT version, compute capability) before loading engines | |

The TensorRT version is **not a build parameter** — each builder image is
prebaked with exactly one. Your `uv.lock` pin picks which builder image
`trtc launch` starts, and `trtc submit` refuses a builder whose baked
TensorRT doesn't match your pin. The manifest records what engines were
actually built with; the runtime enforces it. `trtc` itself declares no
dependencies.

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
a default spec: strongly typed, TensorRT defaults, no profiles), then point
submit (or a local `trtc-server build`) at it:

```sh
uv run trtc submit model.onnx --builder http://builder:8080 --out .
uv run trtc-server build model.onnx      # same thing, locally on a GPU box
```

## The builder

`ghcr.io/dialohq/trtc-builder` — built by CI from the flake
(`nix build .#trtc-builder`, x2container). It is a **fixed, correct
environment**: trtc plus one exact TensorRT, and nothing resolved at runtime.
Like a nix derivation, the image is the pin; to build for another TensorRT
you run the image built for that version. No `uv run --with`, no PATH tricks
— the venv is either correct or the build fails.

### Launch one on vast.ai

`nix run .#launch-builder` rents a GPU, starts the builder image matching
your lock's TensorRT pin, waits until it answers, and prints the URL to point
`trtc` at:

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
(and any external weight data files) next to it, exactly the on-disk layout —
returning one engine. No build parameters in the request. Multi-component
models are composed client-side. Engine + timing caches persist under
`TRTC_DATA_DIR`, so any HTTP client works:

```sh
tar cf job.tar trtc_build_spec.json model.onnx model.onnx.data
curl -X POST -H 'Content-Type: application/x-tar' \
    --data-binary @job.tar "http://builder:8080/builds"
```

```sh
# vast.ai
vastai create instance <offer> --image ghcr.io/dialohq/trtc-builder:trt11.1 \
    --disk 40 --env '-p 8080:8080 -e TRTC_TOKEN=...'

# anywhere with a GPU
docker run --gpus all -p 8080:8080 -v trtc-data:/data ghcr.io/dialohq/trtc-builder
```

Set `TRTC_TOKEN` for auth, `TRTC_IDLE_TIMEOUT` for self-shutdown on idle.
`trtc inspect <dir>` pretty-prints specs/manifests; `trtc info` shows local
GPU + TRT facts; `GET /info` shows the builder's baked TensorRT.
