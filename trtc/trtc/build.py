"""Build stage: plan.json + ONNX -> TensorRT engines + manifest.json.

Runs on hardware matching the deployment GPU, in an environment whose
tensorrt-cu12 matches the plan's pin. Needs no torch and no model code.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from .plan import (
    MANIFEST_FILE,
    query_gpu,
    sha256_file,
    trt_pin_satisfied,
    write_json,
)


def _installed_trt_version(trt: Any) -> str:
    return getattr(trt, "__version__", "unknown")


def _check_trt_version(trt: Any, plan: dict[str, Any]) -> None:
    pinned = plan["tensorrt_version"]
    installed = _installed_trt_version(trt)
    # Full numeric comparison: a '.postN' wheel suffix is ignored, but a
    # different minor (e.g. installed 10.1 vs pinned 10.13.x) is rejected.
    # The environment must be correct; there is no override.
    if trt_pin_satisfied(pinned, installed):
        return
    raise RuntimeError(
        f"This environment has tensorrt {installed} but the plan pins {pinned}. "
        f"Build in an environment whose tensorrt-cu12 is {pinned} (pin it in the "
        "project's uv.lock and sync, or use a builder image built for that version)."
    )


def _engine_cache_key(component: dict[str, Any], trt_version: str, compute_capability: str | None) -> str:
    identity = json.dumps(
        {
            "onnx": component["onnx_sha256"],
            "profiles": component["profiles"],
            "dtype": component["dtype"],
            "workspace": component["workspace_bytes"],
            "strongly_typed": component.get("strongly_typed", True),
            "trt": trt_version,
            "cc": compute_capability,
        },
        sort_keys=True,
    )
    return hashlib.sha256(identity.encode()).hexdigest()


def _build_engine(
    trt: Any,
    onnx_path: Path,
    engine_path: Path,
    *,
    workspace_bytes: int,
    profiles: dict[str, dict[str, list[int]]],
    strongly_typed: bool,
    timing_cache: Any | None,
) -> None:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED) if strongly_typed else 0
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT failed to parse {onnx_path}:\n{errors}")

    config = builder.create_builder_config()
    if hasattr(trt, "MemoryPoolType"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    if timing_cache is not None:
        config.set_timing_cache(timing_cache, ignore_mismatch=False)
    if profiles:
        profile = builder.create_optimization_profile()
        for tensor_name, shapes in profiles.items():
            profile.set_shape(tensor_name, tuple(shapes["min"]), tuple(shapes["opt"]), tuple(shapes["max"]))
        config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT failed to build engine from {onnx_path}")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))


def build_plan(
    plan: dict[str, Any],
    work_dir: str | Path,
    out_dir: str | Path | None = None,
    *,
    force: bool = False,
    timing_cache_path: str | Path | None = None,
    engine_cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build every engine in a plan whose ONNX files live in work_dir."""
    import tensorrt as trt

    work_dir = Path(work_dir)
    out_dir = Path(out_dir) if out_dir is not None else work_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _check_trt_version(trt, plan)

    gpu = query_gpu()
    trt_version = _installed_trt_version(trt)

    engine_cache_dir = Path(engine_cache_dir) if engine_cache_dir else (
        Path(cache) if (cache := os.getenv("TRTC_CACHE_DIR")) else None
    )
    if engine_cache_dir:
        (engine_cache_dir / "engines").mkdir(parents=True, exist_ok=True)

    timing_cache = None
    if timing_cache_path is not None:
        timing_cache_path = Path(timing_cache_path)
        config_for_cache = trt.Builder(trt.Logger(trt.Logger.WARNING)).create_builder_config()
        cache_bytes = timing_cache_path.read_bytes() if timing_cache_path.exists() else b""
        timing_cache = config_for_cache.create_timing_cache(cache_bytes)

    built_components = []
    for component in plan["components"]:
        onnx_path = work_dir / component["onnx"]
        engine_path = out_dir / component["engine"]
        if not onnx_path.exists():
            raise FileNotFoundError(f"Plan references missing ONNX file: {onnx_path}")

        cache_key = _engine_cache_key(component, trt_version, gpu["compute_capability"])
        cached_engine = (engine_cache_dir / "engines" / f"{cache_key}.engine") if engine_cache_dir else None
        # Sidecar recording which cache key produced the engine at engine_path,
        # so an existing engine is only reused when it matches THIS plan (same
        # ONNX hash, profiles, dtype, TRT, arch) — never a stale one.
        key_path = engine_path.with_name(engine_path.name + ".key")

        def _record_key() -> None:
            key_path.write_text(cache_key)

        if not force and engine_path.exists() and key_path.exists() and key_path.read_text() == cache_key:
            print(f"keep existing {engine_path} ({cache_key[:12]})")
        elif not force and cached_engine is not None and cached_engine.exists():
            print(f"cache hit {component['name']} ({cache_key[:12]})")
            engine_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cached_engine, engine_path)
            _record_key()
        else:
            print(f"build {component['name']} -> {engine_path}")
            started = time.monotonic()
            _build_engine(
                trt,
                onnx_path,
                engine_path,
                workspace_bytes=int(component["workspace_bytes"]),
                profiles=component["profiles"],
                strongly_typed=bool(component.get("strongly_typed", True)),
                timing_cache=timing_cache,
            )
            print(f"built {component['name']} in {time.monotonic() - started:.1f}s")
            _record_key()
            if cached_engine is not None:
                shutil.copyfile(engine_path, cached_engine)

        built_components.append(
            {
                **component,
                "engine_sha256": sha256_file(engine_path),
                "engine_size": engine_path.stat().st_size,
            }
        )

    if timing_cache is not None and timing_cache_path is not None:
        serialized_cache = timing_cache.serialize()
        if serialized_cache:
            timing_cache_path.parent.mkdir(parents=True, exist_ok=True)
            timing_cache_path.write_bytes(bytes(serialized_cache))

    manifest = {
        **plan,
        "components": built_components,
        "build": {
            "tensorrt_version": trt_version,
            "gpu_name": gpu["gpu_name"],
            "compute_capability": gpu["compute_capability"],
            "driver_version": gpu["driver_version"],
            "used_timing_cache": timing_cache is not None,
        },
    }
    write_json(out_dir / MANIFEST_FILE, manifest)
    return manifest
