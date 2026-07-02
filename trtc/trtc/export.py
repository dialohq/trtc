"""Export stage: bundle declaration -> ONNX files + plan.json.

Runs where the model code lives, with the project's own torch. This is the
only stage that imports the model; the ONNX+plan directory it produces is
self-contained build input for any builder.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

from .plan import PLAN_FILE, component_record, make_plan, sha256_file, write_json
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


def export_bundle(
    bundle: Bundle,
    out_dir: str | Path,
    *,
    device: str = "cuda",
    tensorrt_version: str,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import torch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    components = []
    for component in bundle.components:
        print(f"export {component.name} -> {out_dir / component.onnx_name}")
        onnx_path = export_component(component, out_dir, device=device)
        components.append(
            component_record(
                name=component.name,
                onnx=component.onnx_name,
                engine=component.engine_name,
                dtype=component.dtype,
                opset=component.opset,
                workspace_gb=component.workspace_gb,
                strongly_typed=component.strongly_typed,
                profiles=component.profiles(),
                onnx_sha256=sha256_file(onnx_path),
                meta=component.meta,
            )
        )

    plan = make_plan(
        bundle=bundle.name,
        tensorrt_version=tensorrt_version,
        components=components,
        engine_dir_hint=bundle.engine_dir_hint,
        meta=bundle.meta,
        provenance={"torch_version": torch.__version__, **(provenance or {})},
    )
    write_json(out_dir / PLAN_FILE, plan)
    return plan
