# trtc-server (C++)

The builder. Plain C++, no Python anywhere near the image:

- **`trtc-server`** — self-contained: bare invocation (or `serve`) runs the
  HTTP job broker; `trtc-server build <target>` builds engines. Broker jobs
  re-exec this same binary's `build` subcommand — a TensorRT crash kills the
  job, never the server.
- **`trtc-build`** — the same build, as its own standalone binary.

Libraries: cpp-httplib, nlohmann_json, OpenSSL, libarchive. TensorRT comes
from NVIDIA's official tarball, hash-pinned in the flake (see below).

## The build spec

Every build option lives in `trtc_build_spec.json`, sitting next to the ONNX
it references. Nothing travels as CLI or query parameters:

```json
{
  "trtc_build_spec": 1,
  "components": [
    {
      "onnx": "model.onnx",
      "strongly_typed": true,
      "profiles": [
        {"x": {"min": [1, 8], "opt": [8, 128], "max": [32, 256]}}
      ],
      "builder_config": { "flags": ["TF32"], "memory_pool_limits": {"WORKSPACE": "8G"} },
      "onnx_sha256": "…optional; the build refuses a mismatching file…",
      "external_data": {"model.onnx.data": null}
    }
  ]
}
```

- `strongly_typed` (default true) — the network creation flag; precision
  comes from the ONNX. TensorRT 11 is strongly-typed only.
- `profiles` — a list of optimization profiles; each entry maps every dynamic
  input to min/opt/max shapes. Several entries build a multi-profile engine.
- `external_data` — the external weight files of >2GB models (file name →
  sha256, or null to skip verification). They live next to the ONNX and
  travel with it in every job.

### builder_config: the entire tensorrt.IBuilderConfig

The vocabulary and its application live in
[`builder_config.hpp`](builder_config.hpp) — explicit tables derived from the
NvInfer headers of every pinned TensorRT, version-guarded where NVIDIA
removed things. Unknown option keys or enum members fail loudly, listing
exactly what this TensorRT has.

| key | value |
|---|---|
| `flags` | list of `BuilderFlag` names, e.g. `["TF32", "SPARSE_WEIGHTS"]` (TRT 10 also has `FP16`, `INT8`, … — gone in 11, strongly-typed only) |
| `memory_pool_limits` | `MemoryPoolType` name → bytes or `"4G"`-style string |
| `builder_optimization_level` | int (0–5) |
| `avg_timing_iterations` | int |
| `max_aux_streams` | int |
| `max_num_tactics` | int |
| `dla_core` | int |
| `default_device_type` | `GPU` \| `DLA` |
| `engine_capability` | `STANDARD` \| `SAFETY` \| `DLA_STANDALONE` |
| `hardware_compatibility_level` | `NONE` \| `AMPERE_PLUS` \| `SAME_COMPUTE_CAPABILITY` |
| `profiling_verbosity` | `NONE` \| `LAYER_NAMES_ONLY` \| `DETAILED` |
| `runtime_platform` | `SAME_AS_BUILD` (the Windows cross resource is not shipped) |
| `tiling_optimization_level` | `NONE` \| `FAST` \| `MODERATE` \| `FULL` |
| `l2_limit_for_tiling` | bytes or `"512M"` |
| `tactic_sources` | list of `TacticSource` names |
| `preview_features` | `PreviewFeature` name → bool |
| `quantization_flags` | TRT 10 only |

Options that are live Python/C++ objects in the API (`int8_calibrator`,
`algorithm_selector`, `progress_monitor`) are not data and cannot appear in
a spec.

## The HTTP API

One job = one tar: the spec + the ONNX (+ external data files), the exact
on-disk layout. Bearer auth when `TRTC_TOKEN` is set.

```
POST /builds[?output_url=<presigned PUT>]   body: job tar, or JSON {"input_url": …}
GET  /builds/{id}[?log_offset=N]            status + incremental log; "result"
                                            holds the manifest once succeeded
GET  /builds/{id}/artifacts                 the engine, raw bytes
GET  /info                                  GPU facts, trtc + TensorRT versions
```

```sh
tar cf job.tar trtc_build_spec.json model.onnx model.onnx.data
curl -X POST -H 'Content-Type: application/x-tar' \
    --data-binary @job.tar "http://builder:8080/builds"
```

Validation is two-layered, deliberately:

1. **Structure, at POST time** (`server.cpp: extract_job_tar`): members must
   be regular files with safe single-segment names; the spec must exist, be
   version 1, single-component, and every file it references must be in the
   tar. Violations are a 400 with the reason.
2. **Options, at job start** (`builder_config.hpp`): only the TensorRT-linked
   build knows what its TensorRT supports, so the broker never second-guesses
   it — a bad option fails the job within a second, with the known-name list
   in the job log and status.

Engines and TensorRT timing caches persist under `TRTC_DATA_DIR`; identical
jobs (same file hashes + spec + TensorRT + GPU arch) are served from the
engine cache.

Environment: `TRTC_TOKEN`, `TRTC_DATA_DIR` (default `~/.cache/trtc`),
`TRTC_IDLE_TIMEOUT` (self-exit after N idle seconds), `TRTC_TENSORRT_VERSION`
(baked by the image, reported by `/info` — the client refuses a builder that
doesn't match its lock pin), `TRTC_BUILD_EXE` (test override for the build
subprocess).

## Building and running

```sh
nix build .#trtc-broker              # broker only, no TensorRT — fast, for tests
nix build .#trtc-server-trt10-13     # trtc-server + trtc-build against TRT 10.13
nix build .#trtc-builder-trt10-13    # the container image

# one-shot local build from loose files (also works unquoted):
nix run github:dialohq/trtc#build-10.13 -- spec.json model.onnx model.onnx.data --out ./engines
```

No environment is ever needed: TensorRT, libcudart, and the driver-injection
locations are baked into RUNPATHs — `env -i ./result/bin/trtc-build <dir>`
builds a real engine on a GPU box.

End-to-end tests drive the real broker binary through the real Python client
with a stubbed build subprocess (no GPU needed):

```sh
TRTC_SERVER_BIN=$PWD/result/bin/trtc-server \
    uv run --package trtc python -m unittest discover -s trtc-server/tests -v
```

## The image

`ghcr.io/dialohq/trtc-builder` — one binary pair, one TensorRT layer (reused
across pushes), CA certs, and a bare-image survival kit (nsswitch.conf for
glibc DNS, passwd/group, sticky /tmp, /data volume). ~2GB; no shell, no
Python, no LD_LIBRARY_PATH. Runs on stock docker/k8s/vast:

```sh
docker run --gpus all -p 8080:8080 -v trtc-data:/data \
    ghcr.io/dialohq/trtc-builder:trt10.13
```

Tags per supported TensorRT: `trt10.13` (moving latest for the line),
`1.0.0-trt10.13` (this trtc release), `1.0.0-trt10.13-<nix hash>`
(immutable, content-addressed). `nix run .#push-builder-images` builds and
pushes all of them; CI runs it on main.

## Adding a TensorRT version

Add an entry to `tensorrtPins` in [`flake.nix`](../flake.nix) (official
tarball URL + sha256) and build: the compiler and `builder_config.hpp` tell
you exactly which vocabulary entries need version guards. That attrset is
the only version list anywhere.
