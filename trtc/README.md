# trtc

Compile PyTorch models to TensorRT engines. Export runs locally with your
project's torch; the engine build runs on deployment-class hardware with the
TensorRT version your lock pins; the runtime refuses engines built for a
different TensorRT or GPU arch.

## Client and server

Two uv workspace packages, one CLI each:

- **`trtc`** (this directory; module `trtc`) — runs where the model code
  lives: `export`, `submit`, `compile` (= export + submit), `launch`,
  `inspect`, `info`. Zero dependencies, never imports tensorrt. Also carries
  the shared plan/manifest contract (`trtc.plan`) and the inference runtime
  (`trtc.runtime`).
- **`trtc-server`** ([`trtc-server/`](../trtc-server), module `trtc_server`)
  — runs on GPU hardware with the pinned TensorRT: `serve` (the builder HTTP
  API) and `build` (engines from a plan dir or bare ONNX). Depends on `trtc`
  (for the contract) and `tensorrt-cu12`; installing it *is* the builder
  environment — that's exactly what the image does. Never imports torch or
  model code. The server builds, end of story.

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

No bundle, no plan file — point submit (or a local `trtc-server build`) at it:

```sh
uv run trtc submit model.onnx --builder http://builder:8080 --out . \
    --shape input=1x80:8x80:16x80        # min:opt:max per dynamic input
uv run trtc-server build model.onnx      # same thing, locally on a GPU box
```

## The builder

`ghcr.io/dialohq/trtc-builder` — built by CI from the flake
(`nix build .#trtc-builder`, x2container). It is a **fixed, correct
environment**: the `trtc-server` workspace member and its locked dependencies
(trtc + the exact TensorRT the workspace `uv.lock` pins), and nothing
resolved at runtime. Like a nix derivation, the image is pinned to one
TensorRT version; a plan pinning a different version fails the job loudly (you
run a builder image built for that version instead). No `uv run --with`, no
PATH tricks — the venv is either correct or the build fails.

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

Deliberately dumb: one job is one ONNX (raw bytes or a presigned URL) plus
query parameters, returning one engine. Multi-component models are composed
client-side. Engine + timing caches persist under `TRTC_DATA_DIR`, so any HTTP
client works:

```sh
curl -X POST --data-binary @model.onnx \
    "http://builder:8080/builds?trt=10.13.3.9.post1&shape=input%3D1x80:8x80:16x80"
```

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
