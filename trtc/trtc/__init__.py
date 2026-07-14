"""trtc: compile PyTorch models to TensorRT engines from a single declaration.

Declare a Bundle once per model; export (torch -> ONNX + plan), build
(ONNX -> engine, locally or on a remote builder) and serve (manifest-validated
engine runners) are all derived from it.
"""

from .buildspec import BuilderConfig, BuildSpec, ComponentSpec, ShapeRange
from .spec import AffineAxis, Axis, Bundle, Component, T, TensorSpec, load_entry

__all__ = [
    "AffineAxis",
    "Axis",
    "BuildSpec",
    "BuilderConfig",
    "Bundle",
    "Component",
    "ComponentSpec",
    "ShapeRange",
    "T",
    "TensorSpec",
    "load_entry",
]
