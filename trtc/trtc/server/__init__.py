"""Server side of trtc: runs on GPU hardware with the pinned TensorRT.

Builds engines from ONNX + plan input and serves the build API. Never imports
torch or model code — export is the client's job (`trtc.client`).
"""
