"""Declarative vocabulary for describing compilable model components.

A model package declares, once, what gets compiled: which submodules, their
named input tensors, and how each dynamic dimension ranges. Everything else
(ONNX export arguments, TensorRT optimization profiles, runtime shape
bucketing metadata) is derived from this single declaration.

This module has no torch or tensorrt imports so declarations stay importable
in any environment.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ContextManager, Mapping, Sequence

PROFILE_KINDS = ("min", "opt", "max")


@dataclass(frozen=True)
class Axis:
    """A named dynamic dimension with a min/opt/max optimization profile."""

    name: str
    min: int
    opt: int
    max: int

    def __post_init__(self) -> None:
        if not 1 <= self.min <= self.opt <= self.max:
            raise ValueError(
                f"Axis {self.name!r} requires 1 <= min <= opt <= max, got ({self.min}, {self.opt}, {self.max})"
            )

    def affine(self, *, scale: int = 1, offset: int = 0, name: str | None = None) -> "AffineAxis":
        """A dimension coupled to this axis: value = scale * axis + offset."""
        return AffineAxis(
            axis=self,
            scale=scale,
            offset=offset,
            name=name or f"{self.name}_x{scale}p{offset}",
        )

    def value(self, kind: str) -> int:
        return int(getattr(self, kind))


@dataclass(frozen=True)
class AffineAxis:
    axis: Axis
    scale: int
    offset: int
    name: str

    def value(self, kind: str) -> int:
        return self.scale * self.axis.value(kind) + self.offset


Dim = int | Axis | AffineAxis


@dataclass(frozen=True)
class TensorSpec:
    """One named input tensor: dims are ints (static) or axes (dynamic).

    dtype: torch dtype name ("float16", "int64", ...); None inherits the
    component dtype. example: optional callable (shape, device, dtype) ->
    torch.Tensor for inputs whose export tracing needs realistic values.
    """

    dims: tuple[Dim, ...]
    dtype: str | None = None
    example: Callable[..., Any] | None = None

    def shape(self, kind: str) -> tuple[int, ...]:
        return tuple(d if isinstance(d, int) else d.value(kind) for d in self.dims)

    def dynamic_axes(self) -> dict[int, str]:
        return {i: d.name for i, d in enumerate(self.dims) if not isinstance(d, int)}


def T(
    dims: Sequence[Dim],
    dtype: str | None = None,
    example: Callable[..., Any] | None = None,
) -> TensorSpec:
    return TensorSpec(dims=tuple(dims), dtype=dtype, example=example)


@dataclass
class Component:
    """One engine to build: a module factory plus its I/O contract."""

    name: str
    module: Callable[[], Any]
    inputs: Mapping[str, TensorSpec]
    outputs: Sequence[str]
    file_stem: str | None = None
    dtype: str = "float32"
    opset: int = 20
    workspace_gb: float = 4.0
    strongly_typed: bool = True
    # Extra dynamic-axis names for outputs, e.g. {"audio": {0: "batch", 2: "audio_samples"}}
    output_axes: Mapping[str, Mapping[int, str]] | None = None
    # Context manager entered around torch.onnx.export (model-owned graph
    # rewrites hide behind this; the tool never knows what it does).
    export_context: Callable[[], ContextManager[Any]] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def stem(self) -> str:
        return self.file_stem or self.name

    @property
    def onnx_name(self) -> str:
        return f"{self.stem}.onnx"

    @property
    def engine_name(self) -> str:
        return f"{self.stem}.engine"

    def profiles(self) -> dict[str, dict[str, list[int]]]:
        return {
            tensor_name: {kind: list(ts.shape(kind)) for kind in PROFILE_KINDS}
            for tensor_name, ts in self.inputs.items()
        }

    def dynamic_axes(self) -> dict[str, dict[int, str]]:
        axes: dict[str, dict[int, str]] = {}
        for tensor_name, ts in self.inputs.items():
            dynamic = ts.dynamic_axes()
            if dynamic:
                axes[tensor_name] = dynamic
        for tensor_name, dims in (self.output_axes or {}).items():
            axes[tensor_name] = {int(i): name for i, name in dims.items()}
        return axes


@dataclass
class Bundle:
    """Everything one model compiles to: components plus optional hooks."""

    name: str
    components: list[Component]
    # Where engines should land by default (e.g. "<model_dir>/tensorrt").
    engine_dir_hint: str | None = None
    # Called after engines exist locally: finalize(manifest_dict, engine_dir).
    finalize: Callable[[dict, Path], None] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def _split_entry(entry: str) -> tuple[str, str]:
    """Split 'target[:attr]' into (target, attr). The ':attr' is only taken
    when it is a bare identifier, so a Windows drive or '.py:func' still work."""
    target, sep, maybe_attr = entry.rpartition(":")
    if sep and maybe_attr.isidentifier():
        return target, maybe_attr
    return entry, "bundle"


def _module_name_for_file(path: Path) -> tuple[str, Path] | None:
    """If path sits inside a package (has __init__.py ancestors), return its
    dotted module name and the directory to put on sys.path so that importing
    it makes relative imports inside the module work."""
    parts = [path.stem]
    directory = path.parent
    while (directory / "__init__.py").exists():
        parts.append(directory.name)
        directory = directory.parent
    if len(parts) == 1:
        return None  # not inside a package
    return ".".join(reversed(parts)), directory


def load_entry(entry: str) -> Callable[..., Bundle]:
    """Resolve a bundle factory from 'path/to/file.py[:attr]' or 'pkg.module:attr'.

    Dispatch is on what the target actually is: an existing .py file (or a
    path ending in .py) loads as a file; otherwise it imports as a module. A
    file inside a package is imported by its dotted name so that relative
    imports in the bundle module resolve."""
    target, attr = _split_entry(entry)
    path = Path(target)
    is_file = path.suffix == ".py" or (path.exists() and path.is_file())

    if is_file:
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Bundle entry not found: {path}")
        resolved = _module_name_for_file(path)
        if resolved is not None:
            module_name, sys_path_dir = resolved
            if str(sys_path_dir) not in sys.path:
                sys.path.insert(0, str(sys_path_dir))
            module = importlib.import_module(module_name)
        else:
            module_name = f"_trtc_entry_{path.stem}"
            module_spec = importlib.util.spec_from_file_location(module_name, path)
            if module_spec is None or module_spec.loader is None:
                raise ImportError(f"Cannot load bundle entry: {path}")
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[module_name] = module
            module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)

    factory = getattr(module, attr, None)
    if factory is None or not callable(factory):
        raise AttributeError(f"Bundle entry {entry!r} has no callable {attr!r}")
    return factory


def parse_option_value(raw: str) -> Any:
    if "," in raw:
        return [parse_option_value(part) for part in raw.split(",") if part != ""]
    lowered = raw.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def parse_options(pairs: Sequence[str]) -> dict[str, Any]:
    options: dict[str, Any] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise ValueError(f"Expected key=value, got {pair!r}")
        options[key.strip()] = parse_option_value(value)
    return options
