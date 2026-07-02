"""Serve stage: load manifest-validated engines and run them on torch tensors.

Generic over models: an EngineRunner binds by tensor name, so anything the
export stage declared is what the runtime binds. Validation compares the
manifest's build facts (TensorRT version, compute capability) against the
current environment before any engine is deserialized.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Mapping

from .plan import read_manifest, trt_versions_compatible

_RUNNER_CACHE: dict[str, "EngineRunner"] = {}
_RUNNER_CACHE_LOCK = threading.Lock()
_VALIDATED_DIRS: set[str] = set()


def clear_runner_cache() -> None:
    with _RUNNER_CACHE_LOCK:
        _RUNNER_CACHE.clear()
        _VALIDATED_DIRS.clear()


def validate_engine_dir(engine_dir: str | Path) -> None:
    """Fail fast if engines in this directory were built for another environment."""
    key = str(Path(engine_dir).expanduser().resolve())
    if key in _VALIDATED_DIRS or os.getenv("TRTC_SKIP_VALIDATION") == "1":
        return
    manifest = read_manifest(Path(key))
    if manifest is None:
        # Pre-manifest engine dirs keep working; there is just nothing to check.
        _VALIDATED_DIRS.add(key)
        return

    import tensorrt as trt

    build = manifest.get("build", {})
    built_with = build.get("tensorrt_version")
    installed = getattr(trt, "__version__", "unknown")
    if built_with and not trt_versions_compatible(built_with, installed):
        raise RuntimeError(
            f"Engines in {key} were built with TensorRT {built_with} but this environment "
            f"runs {installed}. Rebuild the engines or align the tensorrt-cu12 pin. "
            "(Set TRTC_SKIP_VALIDATION=1 to bypass.)"
        )

    built_cc = build.get("compute_capability")
    if built_cc:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            current_cc = f"{major}.{minor}"
            if current_cc != built_cc:
                raise RuntimeError(
                    f"Engines in {key} were built for compute capability {built_cc} "
                    f"({build.get('gpu_name')}) but this GPU is {current_cc}. "
                    "Rebuild on matching hardware. (Set TRTC_SKIP_VALIDATION=1 to bypass.)"
                )
    _VALIDATED_DIRS.add(key)


def _torch_dtype_from_trt(trt: Any, torch: Any, dtype: Any) -> Any:
    if dtype == trt.DataType.FLOAT:
        return torch.float32
    if dtype == trt.DataType.HALF:
        return torch.float16
    if dtype == trt.DataType.INT32:
        return torch.int32
    if dtype == trt.DataType.INT64:
        return torch.int64
    if dtype == trt.DataType.BOOL:
        return torch.bool
    raise TypeError(f"Unsupported TensorRT dtype: {dtype}")


class EngineRunner:
    """Deserialized engine + execution context; binds torch CUDA tensors by name."""

    def __init__(self, engine_path: str | Path, *, name: str | None = None) -> None:
        self.engine_path = Path(engine_path)
        self.name = name or self.engine_path.stem
        if not self.engine_path.exists():
            raise FileNotFoundError(f"Missing TensorRT engine: {self.engine_path}")

        import tensorrt as trt
        import torch

        self._trt = trt
        self._torch = torch

        self._logger = trt.Logger(trt.Logger.WARNING)
        with trt.Runtime(self._logger) as runtime:
            self._engine = runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {self.engine_path}")

        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError(f"Failed to create TensorRT execution context: {self.engine_path}")
        self._streams: dict[int, Any] = {}
        self._run_lock = threading.Lock()

        self._input_names: list[str] = []
        self._output_names: list[str] = []
        for index in range(self._engine.num_io_tensors):
            tensor_name = self._engine.get_tensor_name(index)
            mode = self._engine.get_tensor_mode(tensor_name)
            if mode == trt.TensorIOMode.INPUT:
                self._input_names.append(tensor_name)
            elif mode == trt.TensorIOMode.OUTPUT:
                self._output_names.append(tensor_name)

    @property
    def input_names(self) -> tuple[str, ...]:
        return tuple(self._input_names)

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(self._output_names)

    def run(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        with self._run_lock:
            return self._run_without_lock(inputs)

    def _run_without_lock(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        torch = self._torch
        if not inputs:
            raise ValueError(f"TensorRT runner {self.name!r} requires input tensors")

        first_input = next(iter(inputs.values()))
        if not first_input.is_cuda:
            raise ValueError(f"TensorRT runner {self.name!r} inputs must be CUDA tensors")

        bound_inputs: dict[str, Any] = {}
        for tensor_name in self._input_names:
            if tensor_name not in inputs:
                raise KeyError(f"Missing TensorRT input for runner {self.name!r}: {tensor_name}")
            tensor = inputs[tensor_name].contiguous()
            expected_dtype = _torch_dtype_from_trt(self._trt, torch, self._engine.get_tensor_dtype(tensor_name))
            if tensor.dtype != expected_dtype:
                tensor = tensor.to(dtype=expected_dtype)
            self._context.set_input_shape(tensor_name, tuple(tensor.shape))
            bound_inputs[tensor_name] = tensor

        outputs: dict[str, Any] = {}
        for tensor_name in self._output_names:
            shape = tuple(self._context.get_tensor_shape(tensor_name))
            if any(dim < 0 for dim in shape):
                self._context.infer_shapes()
                shape = tuple(self._context.get_tensor_shape(tensor_name))
            dtype = _torch_dtype_from_trt(self._trt, torch, self._engine.get_tensor_dtype(tensor_name))
            outputs[tensor_name] = torch.empty(shape, device=first_input.device, dtype=dtype)

        for tensor_name, tensor in bound_inputs.items():
            self._context.set_tensor_address(tensor_name, tensor.data_ptr())
        for tensor_name, tensor in outputs.items():
            self._context.set_tensor_address(tensor_name, tensor.data_ptr())

        device_index = first_input.device.index if first_input.device.index is not None else torch.cuda.current_device()
        stream = self._streams.get(device_index)
        if stream is None:
            stream = torch.cuda.Stream(device=first_input.device)
            self._streams[device_index] = stream

        current_stream = torch.cuda.current_stream(first_input.device)
        stream.wait_stream(current_stream)
        with torch.cuda.stream(stream):
            if not self._context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError(f"TensorRT execution failed for runner {self.name!r}: {self.engine_path}")
        current_stream.wait_stream(stream)
        return outputs


def get_engine_runner(engine_path: str | Path, *, name: str | None = None, share: bool = True) -> EngineRunner:
    engine_path = Path(engine_path).expanduser().resolve()
    validate_engine_dir(engine_path.parent)
    if not share:
        return EngineRunner(engine_path, name=name)
    key = str(engine_path)
    with _RUNNER_CACHE_LOCK:
        runner = _RUNNER_CACHE.get(key)
        if runner is None:
            runner = EngineRunner(engine_path, name=name)
            _RUNNER_CACHE[key] = runner
        return runner
