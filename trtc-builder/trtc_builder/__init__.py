"""Deployment shim for the trtc builder image.

Declares the builder's fixed environment (trtc + pinned TensorRT); the server
entrypoint lives in trtc.cli (`trtc serve`).
"""
