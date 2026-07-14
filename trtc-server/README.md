# trtc-server

The build side of trtc: `trtc-server serve` runs the builder HTTP API,
`trtc-server build` builds engines from a plan dir or bare ONNX. Installing
this package is the whole builder environment — `trtc-plan` (the shared
plan/manifest contract) plus the pinned TensorRT come with it; the builder
image is exactly `uv`-installing this member. No torch, no model code, no
client code.
