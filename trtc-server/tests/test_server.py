from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

# The client package is a real dependency of trtc-server (for trtc.plan), so
# these tests drive the server through the actual client.
from trtc.client.remote import download_engine, submit_build, wait_for_build
from trtc_server.app import BuilderState, make_handler
from trtc_server.build import _engine_cache_key


class EngineCacheKeyTests(unittest.TestCase):
    def test_engine_cache_key_tracks_identity(self):
        component = {
            "onnx_sha256": "abc",
            "profiles": {"x": {"min": [1], "opt": [2], "max": [4]}},
            "dtype": "float16",
            "workspace_bytes": 1024,
            "strongly_typed": True,
        }
        key = _engine_cache_key(component, "10.13.3.9", "8.9")
        self.assertEqual(key, _engine_cache_key(dict(component), "10.13.3.9", "8.9"))
        self.assertNotEqual(key, _engine_cache_key(component, "10.13.3.9", "9.0"))
        self.assertNotEqual(key, _engine_cache_key({**component, "onnx_sha256": "def"}, "10.13.3.9", "8.9"))


class ServerRoundTripTests(unittest.TestCase):
    """Raw ONNX + query params up -> stubbed build -> engine bytes down."""

    RESULT = {
        "components": [{"name": "m", "engine": "m.engine", "engine_sha256": "abc", "engine_size": 9}],
        "build": {"tensorrt_version": "10.13.3.9.post1", "compute_capability": "8.9"},
    }

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp_path = Path(self._tmp.name)
        self.onnx = tmp_path / "m.onnx"
        self.onnx.write_bytes(b"fake-onnx-bytes")
        self.state = BuilderState(tmp_path / "data")
        # Stand-in for the real build subprocess: engine = copy of the onnx,
        # plus the single-component manifest trtc-server build would write.
        result_json = json.dumps(self.RESULT).replace('"', '\\"')
        self._build_command = patch(
            "trtc_server.app._build_command",
            lambda state, onnx_path, output_dir, params: [
                "sh", "-c",
                f'cp {onnx_path} {output_dir}/{params["name"]}.engine'
                f' && printf "%s" "{result_json}" > {output_dir}/manifest.json',
            ],
        )
        self._build_command.start()

        handler = make_handler(self.state, token="secret")
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        threading.Thread(target=self._drain_queue, daemon=True).start()

    def _drain_queue(self):
        from trtc_server.app import _run_job

        while True:
            job_id = self.state.queue.get()
            _run_job(self.state, job_id)

    def tearDown(self):
        self.server.shutdown()
        self._build_command.stop()
        self._tmp.cleanup()

    def test_rejects_missing_token(self):
        request = urllib.request.Request(f"{self.url}/info")
        with self.assertRaises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(request)
        self.assertEqual(raised.exception.code, 401)

    def test_rejects_traversal_name(self):
        from trtc_server.app import parse_build_params

        with self.assertRaises(ValueError):
            parse_build_params({"trt": ["10.13.3.9.post1"], "name": ["../../etc/passwd"]})
        with self.assertRaises(ValueError):
            parse_build_params({"trt": ["10.13.3.9.post1"], "name": ["a/b"]})
        # A normal component name is accepted.
        self.assertEqual(
            parse_build_params({"trt": ["10.13.3.9.post1"], "name": ["diffusion_dynamic_s10"]})["name"],
            "diffusion_dynamic_s10",
        )

    def test_rejects_missing_trt_param(self):
        request = urllib.request.Request(
            f"{self.url}/builds", data=b"onnx", method="POST",
            headers={"Authorization": "Bearer secret", "Content-Type": "application/octet-stream"},
        )
        with self.assertRaises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(request)
        self.assertEqual(raised.exception.code, 400)

    def test_full_round_trip(self):
        params = {
            "trt_version": "10.13.3.9.post1",
            "name": "m",
            "dtype": "float16",
            "workspace_gb": 1.0,
            "shapes": ["x=1x3:2x3:4x3"],
        }
        job_id = submit_build(self.url, self.onnx, params=params, token="secret")
        job = wait_for_build(self.url, job_id, token="secret", poll_seconds=0.05, echo_log=False)
        self.assertEqual(job["state"], "succeeded", job.get("error"))
        self.assertEqual(job["result"], self.RESULT)
        self.assertEqual(job["params"]["shapes"], params["shapes"])

        dest = Path(self._tmp.name) / "out" / "m.engine"
        download_engine(self.url, job_id, dest, token="secret")
        self.assertEqual(dest.read_bytes(), b"fake-onnx-bytes")


if __name__ == "__main__":
    unittest.main()
