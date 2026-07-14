"""End-to-end test of the C++ broker: the real binary, the real Python client.

TRTC_SERVER_BIN must point at a compiled trtc-server; the TensorRT build step
is stubbed with a fake trtc-build (no GPU needed), everything else — HTTP,
auth, job lifecycle, log streaming, engine download — is the real thing.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import tempfile
import textwrap
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from trtc.remote import download_engine, submit_build, wait_for_build

SERVER_BIN = os.environ.get("TRTC_SERVER_BIN")

RESULT = {
    "components": [{"name": "m", "engine": "m.engine", "engine_sha256": "abc", "engine_size": 9}],
    "build": {"tensorrt_version": "10.13.3.9", "compute_capability": "8.9"},
}


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@unittest.skipUnless(SERVER_BIN, "TRTC_SERVER_BIN not set")
class ServerE2ETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        cls.onnx = tmp / "m.onnx"
        cls.onnx.write_bytes(b"fake-onnx-bytes")

        # Stand-in for the real trtc-build: engine = copy of the onnx, plus
        # the single-component manifest the real tool would write.
        stub = tmp / "fake-trtc-build"
        stub.write_text(
            textwrap.dedent(
                """\
                #!/bin/sh
                onnx=$1; shift
                while [ $# -gt 0 ]; do
                  case $1 in
                    --name) name=$2; shift 2;;
                    --out) out=$2; shift 2;;
                    *) shift;;
                  esac
                done
                echo "stub build of $name"
                cp "$onnx" "$out/$name.engine"
                cat > "$out/manifest.json" <<'EOF'
                %s
                EOF
                """
            )
            % json.dumps(RESULT)
        )
        stub.chmod(0o755)

        cls.port = _free_port()
        cls.url = f"http://127.0.0.1:{cls.port}"
        cls.server = subprocess.Popen(
            [SERVER_BIN, "serve", "--host", "127.0.0.1", "--port", str(cls.port)],
            env={
                **os.environ,
                "TRTC_DATA_DIR": str(tmp / "data"),
                "TRTC_TOKEN": "secret",
                "TRTC_BUILD_EXE": str(stub),
            },
        )
        for _ in range(100):
            try:
                request = urllib.request.Request(f"{cls.url}/info", headers={"Authorization": "Bearer secret"})
                with urllib.request.urlopen(request, timeout=1):
                    break
            except OSError:
                time.sleep(0.1)
        else:
            raise RuntimeError("trtc-server did not come up")

    @classmethod
    def tearDownClass(cls):
        cls.server.terminate()
        cls.server.wait(timeout=10)
        cls._tmp.cleanup()

    def test_rejects_missing_token(self):
        with self.assertRaises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(f"{self.url}/info")
        self.assertEqual(raised.exception.code, 401)

    def test_info_reports_facts(self):
        request = urllib.request.Request(f"{self.url}/info", headers={"Authorization": "Bearer secret"})
        with urllib.request.urlopen(request) as response:
            info = json.loads(response.read())
        self.assertEqual(
            {"gpu_name", "compute_capability", "driver_version", "trtc", "jobs", "cache_dir"}, set(info)
        )

    def test_rejects_missing_trt_param_and_traversal_name(self):
        for query in ("", "trt=10.13.3.9&name=../../etc/passwd"):
            request = urllib.request.Request(
                f"{self.url}/builds?{query}", data=b"onnx", method="POST",
                headers={"Authorization": "Bearer secret", "Content-Type": "application/octet-stream"},
            )
            with self.assertRaises(urllib.error.HTTPError) as raised:
                urllib.request.urlopen(request)
            self.assertEqual(raised.exception.code, 400)

    def test_unknown_job_is_404(self):
        request = urllib.request.Request(
            f"{self.url}/builds/eeeeeeeeeeee", headers={"Authorization": "Bearer secret"}
        )
        with self.assertRaises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(request)
        self.assertEqual(raised.exception.code, 404)

    def test_full_round_trip(self):
        params = {
            "trt_version": "10.13.3.9",
            "name": "m",
            "dtype": "float16",
            "workspace_gb": 1.0,
            "shapes": ["x=1x3:2x3:4x3"],
        }
        job_id = submit_build(self.url, self.onnx, params=params, token="secret")
        job = wait_for_build(self.url, job_id, token="secret", poll_seconds=0.05, echo_log=False)
        self.assertEqual(job["state"], "succeeded", job.get("error"))
        self.assertEqual(job["result"], RESULT)
        self.assertEqual(job["params"]["shapes"], params["shapes"])
        self.assertIn("stub build of m", job.get("log", "") or self._full_log(job_id))

        dest = Path(self._tmp.name) / "out" / "m.engine"
        download_engine(self.url, job_id, dest, token="secret")
        self.assertEqual(dest.read_bytes(), b"fake-onnx-bytes")

    def _full_log(self, job_id: str) -> str:
        request = urllib.request.Request(
            f"{self.url}/builds/{job_id}", headers={"Authorization": "Bearer secret"}
        )
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read())["log"]


if __name__ == "__main__":
    unittest.main()
