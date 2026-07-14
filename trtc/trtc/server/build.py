"""Build stage: trtc_build_spec.json + ONNX -> TensorRT engines + manifest.json.

Runs on hardware matching the deployment GPU with whatever TensorRT the
environment prebakes — the version is not a build parameter. The manifest
records what the engines were actually built with; the runtime enforces it.
Needs no torch and no model code.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

from ..buildspec import (
    MANIFEST_FILE,
    BuilderConfig,
    BuildSpec,
    ComponentSpec,
    query_gpu,
    sha256_file,
    write_json,
)


def _installed_trt_version(trt: Any) -> str:
    return getattr(trt, "__version__", "unknown")


def _engine_cache_key(
    component: ComponentSpec,
    file_hashes: dict[str, str],
    trt_version: str,
    compute_capability: str | None,
) -> str:
    """file_hashes: actual sha256 of the ONNX and each external data file —
    the verified content, not whatever the spec declared."""
    identity = json.dumps(
        {
            "files": file_hashes,
            "component": {
                k: v for k, v in component.to_dict().items() if k not in ("onnx_sha256", "external_data", "meta")
            },
            "trt": trt_version,
            "cc": compute_capability,
        },
        sort_keys=True,
    )
    return hashlib.sha256(identity.encode()).hexdigest()


def _trt_enum(enum_type: Any, member: str, *, context: str) -> Any:
    value = getattr(enum_type, str(member).upper(), None)
    if value is None:
        known = ", ".join(sorted(name for name in dir(enum_type) if name.isupper()))
        raise ValueError(f"{context}: this TensorRT has no {enum_type.__name__}.{str(member).upper()} (known: {known})")
    return value


def _coerce_attribute(name: str, current: Any, value: Any) -> Any:
    """Coerce a spec value onto an IBuilderConfig attribute: strings resolve
    as enum member names against the attribute's current type."""
    if isinstance(value, str) and not isinstance(current, str):
        return _trt_enum(type(current), value, context=f"builder_config.{name}")
    return value


# BuilderConfig fields that are not plain IBuilderConfig attributes: they map
# to dedicated setter calls instead of setattr.
_STRUCTURED_FIELDS = ("flags", "memory_pool_limits", "quantization_flags", "tactic_sources", "preview_features")


def apply_builder_config(trt: Any, config: Any, builder_config: BuilderConfig) -> None:
    """Apply a spec's BuilderConfig to a live tensorrt.IBuilderConfig.

    Field names map 1:1 onto IBuilderConfig attributes; flags, pool limits,
    quantization flags, tactic sources, and preview features go through their
    dedicated setters. Unknown names fail loudly with what the pinned
    TensorRT actually has."""
    for flag_name in builder_config.flags:
        config.set_flag(_trt_enum(trt.BuilderFlag, flag_name, context="builder_config.flags"))
    for pool_name, limit in builder_config.memory_pool_limits.items():
        pool = _trt_enum(trt.MemoryPoolType, pool_name, context="builder_config.memory_pool_limits")
        config.set_memory_pool_limit(pool, int(limit))
    for flag_name in builder_config.quantization_flags:
        config.set_quantization_flag(_trt_enum(trt.QuantizationFlag, flag_name, context="builder_config.quantization_flags"))
    if builder_config.tactic_sources is not None:
        mask = 0
        for source in builder_config.tactic_sources:
            mask |= 1 << int(_trt_enum(trt.TacticSource, source, context="builder_config.tactic_sources"))
        config.set_tactic_sources(mask)
    for feature_name, enabled in builder_config.preview_features.items():
        feature = _trt_enum(trt.PreviewFeature, feature_name, context="builder_config.preview_features")
        config.set_preview_feature(feature, bool(enabled))

    for spec_field in fields(builder_config):
        if spec_field.name in _STRUCTURED_FIELDS:
            continue
        value = getattr(builder_config, spec_field.name)
        if value is None:
            continue
        if not hasattr(config, spec_field.name):
            raise ValueError(
                f"builder_config.{spec_field.name}: this TensorRT's IBuilderConfig has no such attribute"
            )
        setattr(config, spec_field.name, _coerce_attribute(spec_field.name, getattr(config, spec_field.name), value))


def _build_engine(
    trt: Any,
    onnx_path: Path,
    engine_path: Path,
    *,
    component: ComponentSpec,
    timing_cache: Any | None,
) -> None:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED) if component.strongly_typed else 0
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    # parse_from_file, not parse(bytes): large models keep their weights in
    # external data files next to the ONNX, which the parser resolves
    # relative to the model's path.
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT failed to parse {onnx_path}:\n{errors}")

    config = builder.create_builder_config()
    apply_builder_config(trt, config, component.builder_config)
    if timing_cache is not None:
        config.set_timing_cache(timing_cache, ignore_mismatch=False)
    for profile_shapes in component.profiles:
        profile = builder.create_optimization_profile()
        for tensor_name, ranges in profile_shapes.items():
            profile.set_shape(tensor_name, tuple(ranges.min), tuple(ranges.opt), tuple(ranges.max))
        config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT failed to build engine from {onnx_path}")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))


def build_spec(
    spec: BuildSpec,
    work_dir: str | Path,
    out_dir: str | Path | None = None,
    *,
    force: bool = False,
    timing_cache_path: str | Path | None = None,
    engine_cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build every engine in a spec whose ONNX files live in work_dir."""
    import tensorrt as trt

    work_dir = Path(work_dir)
    out_dir = Path(out_dir) if out_dir is not None else work_dir
    out_dir.mkdir(parents=True, exist_ok=True)

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
    for component in spec.components:
        onnx_path = work_dir / component.onnx
        engine_path = out_dir / component.engine
        declared = {component.onnx: component.onnx_sha256, **component.external_data}
        file_hashes: dict[str, str] = {}
        for file_name, declared_sha in declared.items():
            file_path = work_dir / file_name
            if not file_path.exists():
                raise FileNotFoundError(f"Spec references missing file: {file_path}")
            actual_sha = sha256_file(file_path)
            if declared_sha and declared_sha != actual_sha:
                raise RuntimeError(
                    f"{file_path} does not match the spec: sha256 {actual_sha}, spec says {declared_sha}. "
                    "Re-export or fix the spec."
                )
            file_hashes[file_name] = actual_sha

        cache_key = _engine_cache_key(component, file_hashes, trt_version, gpu["compute_capability"])
        cached_engine = (engine_cache_dir / "engines" / f"{cache_key}.engine") if engine_cache_dir else None
        # Sidecar recording which cache key produced the engine at engine_path,
        # so an existing engine is only reused when it matches THIS spec (same
        # ONNX hash, profiles, builder config, TRT, arch) — never a stale one.
        key_path = engine_path.with_name(engine_path.name + ".key")

        def _record_key() -> None:
            key_path.write_text(cache_key)

        if not force and engine_path.exists() and key_path.exists() and key_path.read_text() == cache_key:
            print(f"keep existing {engine_path} ({cache_key[:12]})")
        elif not force and cached_engine is not None and cached_engine.exists():
            print(f"cache hit {component.name} ({cache_key[:12]})")
            engine_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cached_engine, engine_path)
            _record_key()
        else:
            print(f"build {component.name} -> {engine_path}")
            started = time.monotonic()
            _build_engine(trt, onnx_path, engine_path, component=component, timing_cache=timing_cache)
            print(f"built {component.name} in {time.monotonic() - started:.1f}s")
            _record_key()
            if cached_engine is not None:
                shutil.copyfile(engine_path, cached_engine)

        built = {
            **component.to_dict(),
            "onnx_sha256": file_hashes[component.onnx],
            "engine": component.engine,
            "engine_sha256": sha256_file(engine_path),
            "engine_size": engine_path.stat().st_size,
        }
        if component.external_data:
            built["external_data"] = {name: file_hashes[name] for name in component.external_data}
        built_components.append(built)

    if timing_cache is not None and timing_cache_path is not None:
        serialized_cache = timing_cache.serialize()
        if serialized_cache:
            timing_cache_path.parent.mkdir(parents=True, exist_ok=True)
            timing_cache_path.write_bytes(bytes(serialized_cache))

    manifest = {
        **spec.to_dict(),
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
