# trtc-builder

Deployment package for the trtc builder image. It exists only to pin the
builder's environment — `trtc` plus the TensorRT version the workspace
`uv.lock` locks — so the image is a correct, fixed artifact. Run it with
`trtc serve`.
