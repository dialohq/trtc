"""Build specs and manifests: the serialized contract between the stages.

- trtc_build_spec.json (written by `trtc export` or by hand), sitting next to
  the ONNX files it references: everything a builder needs. The dataclasses
  here map 1:1 onto the TensorRT Python API — BuilderConfig fields are
  tensorrt.IBuilderConfig attributes, enum values are spelled as member names
  — so the JSON is the TensorRT API written down as data. No model code, no
  torch, no tensorrt imports.
- manifest.json (written by build, consumed by the runtime): the build spec
  plus build facts (actual TensorRT version, GPU arch, engine hashes). The
  runtime refuses engines whose build facts don't match its environment.
"""

from __future__ import annotations

import ctypes
import hashlib
import io
import json
import re
import tarfile
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

BUILD_SPEC_FILE = "trtc_build_spec.json"
MANIFEST_FILE = "manifest.json"
BUILD_SPEC_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _reject_unknown_keys(cls_name: str, data: dict[str, Any], known: set[str]) -> None:
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"{cls_name}: unknown key(s) {unknown}; known keys: {sorted(known)}")


_SIZE_SUFFIXES = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}


def parse_size(raw: int | str) -> int:
    """4294967296, '4G', '512M' or '1.5G' -> bytes."""
    if isinstance(raw, int):
        return raw
    raw = raw.strip()
    scale = _SIZE_SUFFIXES.get(raw[-1:].upper())
    if scale is not None:
        return int(float(raw[:-1]) * scale)
    return int(raw)


@dataclass(frozen=True)
class ShapeRange:
    """min/opt/max shapes of one input tensor within one optimization profile."""

    min: list[int]
    opt: list[int]
    max: list[int]

    def __post_init__(self) -> None:
        if len({len(self.min), len(self.opt), len(self.max)}) != 1:
            raise ValueError(f"ShapeRange: min/opt/max must have the same rank, got {self}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ShapeRange":
        _reject_unknown_keys("ShapeRange", data, {"min", "opt", "max"})
        try:
            return cls(min=list(data["min"]), opt=list(data["opt"]), max=list(data["max"]))
        except KeyError as missing:
            raise ValueError(f"ShapeRange: missing key {missing}") from missing


# One optimization profile: input tensor name -> its shape range. An engine
# can carry several profiles; ComponentSpec.profiles is a list of these.
Profile = dict[str, ShapeRange]


@dataclass
class BuilderConfig:
    """tensorrt.IBuilderConfig, 1:1, as data.

    Every field name is the IBuilderConfig attribute it sets; enum values are
    written as member names (e.g. profiling_verbosity="DETAILED"). None (or an
    empty container) means "leave TensorRT's default". Options that are live
    Python objects rather than data (int8_calibrator, algorithm_selector,
    profile_stream, progress_monitor) cannot be expressed in a spec file.

    'flags' are tensorrt.BuilderFlag names applied via set_flag;
    'memory_pool_limits' maps tensorrt.MemoryPoolType names to bytes via
    set_memory_pool_limit (sizes may be written as "4G" strings);
    'quantization_flags' are tensorrt.QuantizationFlag names;
    'tactic_sources' are tensorrt.TacticSource names combined into the mask
    set_tactic_sources takes; 'preview_features' maps tensorrt.PreviewFeature
    names to booleans via set_preview_feature.
    """

    flags: list[str] = field(default_factory=list)
    memory_pool_limits: dict[str, int] = field(default_factory=dict)
    quantization_flags: list[str] = field(default_factory=list)
    tactic_sources: list[str] | None = None
    preview_features: dict[str, bool] = field(default_factory=dict)
    avg_timing_iterations: int | None = None
    builder_optimization_level: int | None = None
    default_device_type: str | None = None
    DLA_core: int | None = None
    engine_capability: str | None = None
    hardware_compatibility_level: str | None = None
    max_aux_streams: int | None = None
    profiling_verbosity: str | None = None
    runtime_platform: str | None = None
    tiling_optimization_level: str | None = None
    l2_limit_for_tiling: int | None = None
    max_num_tactics: int | None = None
    plugins_to_serialize: list[str] | None = None

    def __post_init__(self) -> None:
        self.flags = [str(flag).upper() for flag in self.flags]
        self.memory_pool_limits = {
            str(pool).upper(): parse_size(limit) for pool, limit in self.memory_pool_limits.items()
        }
        self.quantization_flags = [str(flag).upper() for flag in self.quantization_flags]
        if self.tactic_sources is not None:
            self.tactic_sources = [str(source).upper() for source in self.tactic_sources]

    def to_dict(self) -> dict[str, Any]:
        """Only what deviates from TensorRT defaults, so specs stay terse."""
        return {key: value for key, value in asdict(self).items() if value not in (None, [], {})}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BuilderConfig":
        _reject_unknown_keys("BuilderConfig", data, {f.name for f in fields(cls)})
        return cls(**data)


@dataclass
class ComponentSpec:
    """One engine to build: an ONNX file plus how to build it."""

    name: str
    onnx: str
    engine: str = ""
    strongly_typed: bool = True
    profiles: list[Profile] = field(default_factory=list)
    builder_config: BuilderConfig = field(default_factory=BuilderConfig)
    # sha256 of the ONNX; export fills it in, hand-written specs may omit it.
    # When present, build refuses an ONNX that doesn't match.
    onnx_sha256: str | None = None
    dtype: str | None = None
    opset: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.engine:
            self.engine = f"{Path(self.onnx).stem}.engine"

    def to_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "name": self.name,
            "onnx": self.onnx,
            "engine": self.engine,
            "strongly_typed": self.strongly_typed,
            "profiles": [
                {tensor: asdict(ranges) for tensor, ranges in profile.items()} for profile in self.profiles
            ],
            "builder_config": self.builder_config.to_dict(),
            "onnx_sha256": self.onnx_sha256,
            "dtype": self.dtype,
            "meta": dict(self.meta),
        }
        if self.opset is not None:
            record["opset"] = self.opset
        return record

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ComponentSpec":
        _reject_unknown_keys("ComponentSpec", data, {f.name for f in fields(cls)})
        try:
            converted = dict(
                data,
                profiles=[
                    {tensor: ShapeRange.from_dict(ranges) for tensor, ranges in profile.items()}
                    for profile in data.get("profiles", [])
                ],
                builder_config=BuilderConfig.from_dict(data.get("builder_config", {})),
            )
            return cls(**converted)
        except (KeyError, TypeError) as bad:
            raise ValueError(f"ComponentSpec: {bad}") from bad


@dataclass
class BuildSpec:
    """The whole trtc_build_spec.json.

    Deliberately no TensorRT version field: the builder environment is
    prebaked with exactly one TensorRT (the image is the pin), the manifest
    records what the engines were actually built with, and the runtime
    validates that record against its own environment."""

    bundle: str
    components: list[ComponentSpec]
    engine_dir_hint: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trtc_build_spec": BUILD_SPEC_VERSION,
            "bundle": self.bundle,
            "engine_dir_hint": self.engine_dir_hint,
            "components": [component.to_dict() for component in self.components],
            "meta": dict(self.meta),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BuildSpec":
        version = data.get("trtc_build_spec")
        if version != BUILD_SPEC_VERSION:
            raise ValueError(f"Unsupported trtc_build_spec version: {version!r} (expected {BUILD_SPEC_VERSION})")
        payload = {key: value for key, value in data.items() if key != "trtc_build_spec"}
        _reject_unknown_keys("BuildSpec", payload, {f.name for f in fields(cls)})
        try:
            return cls(**dict(payload, components=[ComponentSpec.from_dict(c) for c in payload["components"]]))
        except (KeyError, TypeError) as bad:
            raise ValueError(f"BuildSpec: {bad}") from bad


def read_build_spec(work_dir: str | Path) -> BuildSpec:
    spec_path = Path(work_dir) / BUILD_SPEC_FILE
    if not spec_path.exists():
        raise FileNotFoundError(f"No {BUILD_SPEC_FILE} in {work_dir}")
    return BuildSpec.from_dict(read_json(spec_path))


def write_build_spec(spec: BuildSpec, work_dir: str | Path) -> Path:
    spec_path = Path(work_dir) / BUILD_SPEC_FILE
    write_json(spec_path, spec.to_dict())
    return spec_path


def default_build_spec_for_onnx(onnx_path: Path) -> BuildSpec:
    """A default single-component spec synthesized for a bare ONNX file with
    no trtc_build_spec.json next to it: strongly typed, TensorRT defaults."""
    component = ComponentSpec(
        name=onnx_path.stem,
        onnx=onnx_path.name,
        onnx_sha256=sha256_file(onnx_path),
    )
    return BuildSpec(bundle=component.name, components=[component])


def resolve_build_target(target: str | Path) -> tuple[Path, BuildSpec]:
    """Resolve what `build`/`submit` point at: a directory containing
    trtc_build_spec.json, or a bare .onnx — which uses the spec sitting next
    to it, or a synthesized default spec if there is none."""
    target = Path(target)
    if target.suffix != ".onnx":
        return target, read_build_spec(target)
    if not target.exists():
        raise FileNotFoundError(f"ONNX file not found: {target}")
    onnx_path = target.resolve()
    if (onnx_path.parent / BUILD_SPEC_FILE).exists():
        return onnx_path.parent, read_build_spec(onnx_path.parent)
    return onnx_path.parent, default_build_spec_for_onnx(onnx_path)


def single_component_spec(spec: BuildSpec, component: ComponentSpec) -> BuildSpec:
    """The spec reduced to one component — the unit a builder job takes."""
    return BuildSpec(
        bundle=spec.bundle,
        components=[component],
        meta=dict(spec.meta),
        provenance=dict(spec.provenance),
    )


# Tar members become file names on the builder, so accept only single
# innocuous path segments (no separators, no '..', no leading dot).
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def safe_name(name: str) -> str:
    if not _SAFE_NAME.match(name) or ".." in name:
        raise ValueError(f"invalid name {name!r}: expected [A-Za-z0-9._-], no separators or '..'")
    return name


def pack_job_tar(spec: BuildSpec, work_dir: str | Path) -> bytes:
    """One builder job as a tar: trtc_build_spec.json with the ONNX it
    references next to it — the same layout as on disk."""
    if len(spec.components) != 1:
        raise ValueError(f"A builder job takes exactly one component, got {len(spec.components)}")
    work_dir = Path(work_dir)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        payload = json.dumps(spec.to_dict(), indent=2, sort_keys=True).encode()
        info = tarfile.TarInfo(BUILD_SPEC_FILE)
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
        onnx_path = work_dir / spec.components[0].onnx
        if not onnx_path.exists():
            raise FileNotFoundError(f"Spec references missing ONNX file: {onnx_path}")
        archive.add(onnx_path, arcname=spec.components[0].onnx)
    return buffer.getvalue()


def extract_job_tar(data: bytes, dest_dir: str | Path) -> BuildSpec:
    """Unpack one job tar into dest_dir and return its validated spec.

    Members are written by hand (never tar.extractall) and their names must
    be safe single path segments; the spec must reference exactly one
    component whose name/onnx/engine are safe file names too."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r") as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    raise ValueError(f"job tar member {member.name!r} is not a regular file")
                handle = archive.extractfile(member)
                if handle is None:
                    raise ValueError(f"cannot read job tar member {member.name!r}")
                (dest_dir / safe_name(member.name)).write_bytes(handle.read())
    except tarfile.TarError as error:
        raise ValueError(f"not a valid job tar: {error}") from error

    spec_path = dest_dir / BUILD_SPEC_FILE
    if not spec_path.exists():
        raise ValueError(f"job tar has no {BUILD_SPEC_FILE}")
    spec = BuildSpec.from_dict(read_json(spec_path))
    if len(spec.components) != 1:
        raise ValueError(f"a builder job takes exactly one component, got {len(spec.components)}")
    component = spec.components[0]
    for value in (component.name, component.onnx, component.engine):
        safe_name(value)
    if not (dest_dir / component.onnx).exists():
        raise ValueError(f"job tar spec references {component.onnx!r} but the tar does not contain it")
    return spec


def assemble_manifest(spec: BuildSpec, component_manifests: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge per-component build results (each a single-component manifest)
    back into the spec — multi-component composition is client-side."""
    by_name = {m["components"][0]["name"]: m for m in component_manifests}
    components = []
    build: dict[str, Any] | None = None
    for component in spec.components:
        result = by_name.get(component.name)
        if result is None:
            raise ValueError(f"No build result for component {component.name!r}")
        built = result["components"][0]
        components.append({**component.to_dict(), **{k: built[k] for k in ("engine_sha256", "engine_size")}})
        if build is not None and build != result["build"]:
            raise ValueError("Component engines were built in different environments")
        build = result["build"]
    return {**spec.to_dict(), "components": components, "build": build}


def read_manifest(engine_dir: Path) -> dict[str, Any] | None:
    manifest_path = Path(engine_dir) / MANIFEST_FILE
    if not manifest_path.exists():
        return None
    return read_json(manifest_path)


def tensorrt_version_from_lock(start: str | Path | None = None) -> str | None:
    """The tensorrt-cu12 pin from the nearest uv.lock, walking up from start/cwd.

    The lock is the source of truth regardless of which dependency group or
    extra carries the pin, and regardless of what happens to be installed.
    """
    import tomllib

    current = Path(start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for directory in (current, *current.parents):
        lock_path = directory / "uv.lock"
        if not lock_path.exists():
            continue
        lock = tomllib.loads(lock_path.read_text())
        for package in lock.get("package", []):
            if package.get("name") in ("tensorrt-cu12", "tensorrt"):
                return package.get("version")
        return None  # nearest lock is authoritative; no pin means no pin
    return None


def installed_tensorrt_version() -> str | None:
    """Version of the installed TensorRT distribution, or None if absent."""
    from importlib.metadata import PackageNotFoundError, version

    for distribution in ("tensorrt-cu12", "tensorrt"):
        try:
            return version(distribution)
        except PackageNotFoundError:
            continue
    return None


def resolve_tensorrt_version(explicit: str | None = None, *, project_dir: str | Path | None = None) -> str:
    """The TensorRT version this project expects.

    Resolution order: explicit flag > uv.lock pin > installed distribution.
    Not a build parameter — builders are prebaked with one TensorRT. It picks
    which builder image `trtc launch` starts and lets the client refuse a
    builder baked with something else before submitting.
    """
    if explicit:
        return explicit

    installed = installed_tensorrt_version()
    locked = tensorrt_version_from_lock(project_dir)
    if locked:
        if installed and installed != locked:
            print(f"WARNING: uv.lock pins tensorrt-cu12 {locked} but {installed} is installed; using the lock")
        return locked
    if installed:
        return installed
    raise SystemExit(
        "Cannot determine TensorRT version: no tensorrt-cu12 pin in any uv.lock above "
        "the current directory, none installed, and --trt-version not given. Pin "
        "tensorrt-cu12 in the project (any group works; the lock is what's read)."
    )


# CUdevice_attribute enums; part of the stable driver ABI.
_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76


def nvidia_kernel_module_version(proc_version_text: str) -> str | None:
    """Driver build (e.g. '590.48.01') from /proc/driver/nvidia/version.

    The NVRM line's wording varies (proprietary vs open kernel module), so
    match the version number itself."""
    for line in proc_version_text.splitlines():
        if line.startswith("NVRM"):
            match = re.search(r"\b(\d+\.\d+(?:\.\d+)*)\b", line)
            return match.group(1) if match else None
    return None


def query_gpu() -> dict[str, str | None]:
    """Hardware facts straight from the CUDA driver API.

    ctypes on libcuda.so.1 — the same host-injected library the engine build
    itself binds, found via the same search path, so these facts and the
    build share one provider. No subprocess: host-injected FHS binaries like
    nvidia-smi cannot exec in a base-less image. Degrades to None off-GPU."""
    info: dict[str, str | None] = {"gpu_name": None, "compute_capability": None, "driver_version": None}

    try:
        cuda = ctypes.CDLL("libcuda.so.1")
    except OSError:
        cuda = None
    if cuda is not None and cuda.cuInit(0) == 0:
        device = ctypes.c_int()
        if cuda.cuDeviceGet(ctypes.byref(device), 0) == 0:
            name = ctypes.create_string_buffer(96)
            if cuda.cuDeviceGetName(name, len(name), device) == 0:
                info["gpu_name"] = name.value.decode(errors="replace")
            major, minor = ctypes.c_int(), ctypes.c_int()
            got_major = cuda.cuDeviceGetAttribute(
                ctypes.byref(major), _CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device
            )
            got_minor = cuda.cuDeviceGetAttribute(
                ctypes.byref(minor), _CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device
            )
            if got_major == 0 and got_minor == 0:
                info["compute_capability"] = f"{major.value}.{minor.value}"

    try:
        info["driver_version"] = nvidia_kernel_module_version(
            Path("/proc/driver/nvidia/version").read_text()
        )
    except OSError:
        pass
    return info


def trt_version_tuple(version: str) -> tuple[int, ...]:
    """Numeric dotted prefix of a TensorRT version, dropping local/build
    suffixes: '10.13.3.9.post1' and '10.13.3.9+cuda12' both -> (10, 13, 3, 9)."""
    parts: list[int] = []
    for part in version.split("+", 1)[0].split("."):
        if not part.isdigit():
            break
        parts.append(int(part))
    return tuple(parts)


def trt_pin_satisfied(pinned: str, installed: str) -> bool:
    """The installed distribution builds the engines the spec's pin describes.

    Compares the full numeric version, so a '.postN' wheel suffix is ignored
    but '10.1' is NOT treated as satisfying a '10.13.x' pin."""
    return trt_version_tuple(pinned) == trt_version_tuple(installed)


def trt_versions_compatible(built_with: str, installed: str) -> bool:
    """Engines are portable within the same TensorRT major.minor."""
    return trt_version_tuple(built_with)[:2] == trt_version_tuple(installed)[:2]
