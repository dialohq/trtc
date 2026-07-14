# trtc-plan

The serialized contract between the trtc stages: plan.json (export -> build)
and manifest.json (build -> runtime), plus the TensorRT version and GPU
plumbing both sides need. Zero dependencies — it exists so `trtc` (client)
and `trtc-server` (builder) can share the contract without importing each
other.
