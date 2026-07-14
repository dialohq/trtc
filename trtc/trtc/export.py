"""Export stage: bundle declaration -> ONNX files + trtc_build_spec.json.

Runs where the model code lives, with the project's own torch. This is the
only stage that imports the model; the ONNX+plan directory it produces is
self-contained build input for any builder.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

from .plan import BUILD_SPEC_VERSION, sha256_file, write_build_spec
from .spec import Bundle, Component


def _torch_dtype(torch: Any, name: str) -> Any:
    dtype = getattr(torch, name, None)
    if dtype is None or not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unknown torch dtype name: {name!r}")
    return dtype


def make_example_inputs(component: Component, *, device: str) -> tuple[Any, ...]:
    import torch

    tensors = []
    for name, tensor_spec in component.inputs.items():
        shape = tensor_spec.shape("opt")
        dtype = _torch_dtype(torch, tensor_spec.dtype or component.dtype)
        if tensor_spec.example is not None:
            tensor = tensor_spec.example(shape=shape, device=device, dtype=dtype)
            if tuple(tensor.shape) != shape:
                raise ValueError(
                    f"{component.name}.{name}: example() returned shape {tuple(tensor.shape)}, expected {shape}"
                )
        elif dtype.is_floating_point:
            tensor = torch.randn(shape, device=device, dtype=dtype)
        else:
            tensor = torch.zeros(shape, device=device, dtype=dtype)
        tensors.append(tensor)
    return tuple(tensors)


def _module_device(module: Any, fallback: str) -> str:
    """Where the module's parameters live — the single source of truth for
    example-input placement. Falls back only for parameterless modules."""
    try:
        return str(next(module.parameters()).device)
    except StopIteration:
        return fallback


def export_component(component: Component, out_dir: Path, *, device: str) -> Path:
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / component.onnx_name
    module = component.module()
    # Place example inputs where the module actually is, so the CLI device flag
    # and the bundle's own device choice cannot disagree and crash tracing.
    example_inputs = make_example_inputs(component, device=_module_device(module, device))
    context = component.export_context() if component.export_context is not None else contextlib.nullcontext()
    with context, torch.no_grad():
        torch.onnx.export(
            module,
            example_inputs,
            str(onnx_path),
            input_names=list(component.inputs.keys()),
            output_names=list(component.outputs),
            opset_version=component.opset,
            dynamo=False,
            do_constant_folding=True,
            dynamic_axes=component.dynamic_axes() or None,
        )
    return onnx_path


def _dir_snapshot(directory: Path) -> dict[str, int]:
    """File name -> mtime_ns, for spotting what an export actually wrote."""
    return {p.name: p.stat().st_mtime_ns for p in directory.iterdir() if p.is_file()}


def new_files(before: dict[str, int], after: dict[str, int]) -> list[str]:
    """Files created or rewritten between two snapshots."""
    return sorted(name for name, mtime in after.items() if before.get(name) != mtime)


def export_bundle(
    bundle: Bundle,
    out_dir: str | Path,
    *,
    device: str = "cuda",
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    components = []
    for component in bundle.components:
        print(f"export {component.name} -> {out_dir / component.onnx_name}")
        before = _dir_snapshot(out_dir)
        onnx_path = export_component(component, out_dir, device=device)
        # Anything else the exporter wrote is external weight data the ONNX
        # references (models >2GB store tensors outside the protobuf); it
        # belongs to the component and travels with it.
        external = [name for name in new_files(before, _dir_snapshot(out_dir)) if name != onnx_path.name]
        record: dict[str, Any] = {
            "onnx": component.onnx_name,
            "strongly_typed": component.strongly_typed,
            "profiles": component.profiles(),
            "builder_config": dict(component.builder_config),
            "onnx_sha256": sha256_file(onnx_path),
        }
        if external:
            record["external_data"] = {name: sha256_file(out_dir / name) for name in external}
        components.append(record)

    spec = {"trtc_build_spec": BUILD_SPEC_VERSION, "components": components}
    write_build_spec(spec, out_dir)
    return spec
