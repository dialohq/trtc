"""Build specs and manifests: the serialized contract with the builder.

- trtc_build_spec.json (written by `trtc export` or by hand) sits next to the
  ONNX files it references and carries every build option — the entire
  tensorrt.IBuilderConfig, as JSON (the builder validates option names
  against its own TensorRT and fails loudly on unknowns). One builder job is
  one tar: the spec plus the single ONNX (and any external weight data files)
  it references, the exact on-disk layout.
- manifest.json (written by the builder, consumed by the runtime): the spec
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


def component_name(component: dict[str, Any]) -> str:
    return Path(component["onnx"]).stem


def engine_name(component: dict[str, Any]) -> str:
    return f"{component_name(component)}.engine"


def read_build_spec(work_dir: Path) -> dict[str, Any]:
    spec_path = Path(work_dir) / BUILD_SPEC_FILE
    if not spec_path.exists():
        raise FileNotFoundError(f"No {BUILD_SPEC_FILE} in {work_dir}")
    spec = read_json(spec_path)
    if spec.get("trtc_build_spec") != BUILD_SPEC_VERSION:
        raise ValueError(f"Unsupported trtc_build_spec version in {spec_path}: {spec.get('trtc_build_spec')!r}")
    return spec


def write_build_spec(spec: dict[str, Any], work_dir: Path) -> Path:
    spec_path = Path(work_dir) / BUILD_SPEC_FILE
    write_json(spec_path, spec)
    return spec_path


def default_spec_for_onnx(onnx_path: Path) -> dict[str, Any]:
    """A default single-component spec for a bare ONNX with no spec next to
    it: strongly typed, TensorRT defaults, no profiles. A sibling
    '<model>.onnx.data' (the usual external-weight-data convention for >2GB
    models) is picked up automatically."""
    component: dict[str, Any] = {"onnx": onnx_path.name, "onnx_sha256": sha256_file(onnx_path)}
    external = onnx_path.with_name(f"{onnx_path.name}.data")
    if external.exists():
        component["external_data"] = {external.name: sha256_file(external)}
    return {"trtc_build_spec": BUILD_SPEC_VERSION, "components": [component]}


def resolve_build_target(target: str | Path) -> tuple[Path, dict[str, Any]]:
    """What `submit` points at: a directory containing trtc_build_spec.json,
    or a bare .onnx — which uses the spec next to it, or defaults."""
    target = Path(target)
    if target.suffix != ".onnx":
        return target, read_build_spec(target)
    if not target.exists():
        raise FileNotFoundError(f"ONNX file not found: {target}")
    onnx_path = target.resolve()
    if (onnx_path.parent / BUILD_SPEC_FILE).exists():
        return onnx_path.parent, read_build_spec(onnx_path.parent)
    return onnx_path.parent, default_spec_for_onnx(onnx_path)


def single_component_spec(component: dict[str, Any]) -> dict[str, Any]:
    """A spec of one component — the unit a builder job takes."""
    return {"trtc_build_spec": BUILD_SPEC_VERSION, "components": [component]}


def pack_job_tar(spec: dict[str, Any], work_dir: Path) -> bytes:
    """One builder job as a tar: trtc_build_spec.json with the ONNX (and any
    external weight data files) it references next to it."""
    if len(spec["components"]) != 1:
        raise ValueError(f"A builder job takes exactly one component, got {len(spec['components'])}")
    component = spec["components"][0]
    work_dir = Path(work_dir)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        payload = json.dumps(spec, indent=2, sort_keys=True).encode()
        info = tarfile.TarInfo(BUILD_SPEC_FILE)
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
        for name in (component["onnx"], *component.get("external_data", {})):
            path = work_dir / name
            if not path.exists():
                raise FileNotFoundError(f"Spec references missing file: {path}")
            archive.add(path, arcname=name)
    return buffer.getvalue()


def assemble_manifest(spec: dict[str, Any], component_manifests: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge per-component build results (each a single-component manifest)
    back into the spec — multi-component composition is client-side."""
    by_onnx = {m["components"][0]["onnx"]: m for m in component_manifests}
    components = []
    build: dict[str, Any] | None = None
    for component in spec["components"]:
        result = by_onnx.get(component["onnx"])
        if result is None:
            raise ValueError(f"No build result for component {component['onnx']!r}")
        components.append(result["components"][0])
        if build is not None and build != result["build"]:
            raise ValueError("Component engines were built in different environments")
        build = result["build"]
    return {**spec, "components": components, "build": build}


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
    Not a build parameter — builder images are prebaked with one TensorRT.
    It picks which builder image `trtc launch` starts and lets the client
    refuse a builder baked with something else before submitting.
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
    """The installed TensorRT builds the engines the pin describes. Compares
    major.minor.patch, so wheel-only '.postN'/build suffixes are ignored but
    '10.1' is NOT treated as satisfying a '10.13.x' pin."""
    return trt_version_tuple(pinned)[:3] == trt_version_tuple(installed)[:3]


def trt_versions_compatible(built_with: str, installed: str) -> bool:
    """Engines are portable within the same TensorRT major.minor."""
    return trt_version_tuple(built_with)[:2] == trt_version_tuple(installed)[:2]
