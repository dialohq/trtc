"""Client side of trtc: runs where the model code lives.

Exports models to ONNX + trtc_build_spec.json with the project's own torch, submits the
result to a builder over HTTP, and provisions builders. Never imports
tensorrt — building engines is the server's job (`trtc.server`).
"""
