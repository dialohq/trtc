"""trtc-server: the build side of trtc, on GPU hardware with a pinned TensorRT.

Builds engines from ONNX + plan input and serves the build API. Depends on
trtc only for the shared plan/manifest contract (trtc.plan); never imports
torch or model code — export is the client's job (the `trtc` CLI).
"""
