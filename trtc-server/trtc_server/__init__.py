"""trtc-server: the build side of trtc, on GPU hardware with a pinned TensorRT.

Builds engines from ONNX + plan input and serves the build API. Depends only
on trtc-plan (the shared plan/manifest contract) and tensorrt; never imports
torch, model code, or the trtc client.
"""
