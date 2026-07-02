"""Build plan and manifest: the serialized contract between the three stages.

- plan.json (written by export, consumed by build): everything the builder
  needs, next to the ONNX files. No model code, no torch.
- manifest.json (written by build, consumed by the runtime): the plan plus
  build facts (actual TensorRT version, GPU arch, engine hashes). The runtime
  refuses engines whose build facts don't match its environment.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import re
from pathlib import Path
from typing import Any

PLAN_FILE = "plan.json"
MANIFEST_FILE = "manifest.json"
PLAN_VERSION = 1


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


def component_record(
    *,
    name: str,
    onnx: str,
    dtype: str,
    workspace_gb: float,
    profiles: dict[str, dict[str, list[int]]],
    onnx_sha256: str,
    engine: str | None = None,
    strongly_typed: bool = True,
    opset: int | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One component entry of a plan — the unit `trtc build` consumes."""
    record: dict[str, Any] = {
        "name": name,
        "onnx": onnx,
        "engine": engine or f"{Path(onnx).stem}.engine",
        "dtype": dtype,
        "workspace_bytes": int(workspace_gb * (1 << 30)),
        "strongly_typed": strongly_typed,
        "profiles": profiles,
        "onnx_sha256": onnx_sha256,
        "meta": dict(meta or {}),
    }
    if opset is not None:
        record["opset"] = opset
    return record


def make_plan(
    *,
    bundle: str,
    tensorrt_version: str,
    components: list[dict[str, Any]],
    engine_dir_hint: str | None = None,
    meta: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "trtc_plan": PLAN_VERSION,
        "bundle": bundle,
        "tensorrt_version": tensorrt_version,
        "engine_dir_hint": engine_dir_hint,
        "components": components,
        "meta": dict(meta or {}),
        "provenance": dict(provenance or {}),
    }


def plan_for_onnx(
    onnx_path: Path,
    *,
    tensorrt_version: str,
    name: str | None = None,
    dtype: str = "float32",
    workspace_gb: float = 4.0,
    profiles: dict[str, dict[str, list[int]]] | None = None,
) -> dict[str, Any]:
    """A single-component plan synthesized for a bare ONNX file — lets
    `trtc build`/`trtc submit` consume an ONNX directly, no plan file."""
    record = component_record(
        name=name or onnx_path.stem,
        onnx=onnx_path.name,
        dtype=dtype,
        workspace_gb=workspace_gb,
        profiles=profiles or {},
        onnx_sha256=sha256_file(onnx_path),
    )
    return make_plan(bundle=record["name"], tensorrt_version=tensorrt_version, components=[record])


def shape_specs(profiles: dict[str, dict[str, list[int]]]) -> list[str]:
    """Profiles as `NAME=MIN:OPT:MAX` strings — the CLI/wire encoding."""
    return [
        "{}={}".format(name, ":".join("x".join(str(d) for d in ranges[kind]) for kind in ("min", "opt", "max")))
        for name, ranges in profiles.items()
    ]


def build_params(component: dict[str, Any], tensorrt_version: str) -> dict[str, Any]:
    """One component of a plan as the flat parameter set a builder job takes."""
    return {
        "trt_version": tensorrt_version,
        "name": component["name"],
        "dtype": component["dtype"],
        "workspace_gb": component["workspace_bytes"] / (1 << 30),
        "shapes": shape_specs(component["profiles"]),
    }


def assemble_manifest(plan: dict[str, Any], component_manifests: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge per-component build results (each a single-component manifest)
    back into the plan — multi-component composition is client-side."""
    by_name = {m["components"][0]["name"]: m for m in component_manifests}
    components = []
    build: dict[str, Any] | None = None
    for component in plan["components"]:
        result = by_name.get(component["name"])
        if result is None:
            raise ValueError(f"No build result for component {component['name']!r}")
        built = result["components"][0]
        components.append({**component, **{k: built[k] for k in ("engine_sha256", "engine_size")}})
        if build is not None and build != result["build"]:
            raise ValueError("Component engines were built in different environments")
        build = result["build"]
    return {**plan, "components": components, "build": build}


def read_plan(work_dir: Path) -> dict[str, Any]:
    plan_path = Path(work_dir) / PLAN_FILE
    if not plan_path.exists():
        raise FileNotFoundError(f"No {PLAN_FILE} in {work_dir}")
    plan = read_json(plan_path)
    if plan.get("trtc_plan") != PLAN_VERSION:
        raise ValueError(f"Unsupported plan version in {plan_path}: {plan.get('trtc_plan')!r}")
    return plan


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
    """The TensorRT version engines must be built with.

    Resolution order: explicit flag > uv.lock pin > installed distribution.
    The plan records this so the builder installs exactly the same version.
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
    """The installed distribution builds the engines the plan's pin describes.

    Compares the full numeric version, so a '.postN' wheel suffix is ignored
    but '10.1' is NOT treated as satisfying a '10.13.x' pin."""
    return trt_version_tuple(pinned) == trt_version_tuple(installed)


def trt_versions_compatible(built_with: str, installed: str) -> bool:
    """Engines are portable within the same TensorRT major.minor."""
    return trt_version_tuple(built_with)[:2] == trt_version_tuple(installed)[:2]
