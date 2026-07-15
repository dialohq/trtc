"""End-to-end test of the C++ broker: the real binary, the real Python client.

TRTC_SERVER_BIN must point at a compiled trtc-server; the TensorRT build step
is stubbed with a fake trtc-build (no GPU needed), everything else — the tar
protocol, spec validation, auth, job lifecycle, log streaming, engine
download — is the real thing.
"""

from __future__ import annotations

import io
import json
import os
import socket
import subprocess
import tarfile
import tempfile
import textwrap
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from trtc.plan import pack_job_tar, read_build_spec, single_component_spec, write_build_spec
from trtc.remote import builder_info, download_engine, submit_build, wait_for_build

SERVER_BIN = os.environ.get("TRTC_SERVER_BIN")

RESULT = {
    "components": [{"onnx": "m.onnx", "engine": "m.engine", "engine_sha256": "abc", "engine_size": 9}],
    "build": {"tensorrt_version": "10.13.3", "compute_capability": "8.9"},
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
        cls.work_dir = tmp / "work"
        cls.work_dir.mkdir()
        (cls.work_dir / "m.onnx").write_bytes(b"fake-onnx-bytes")
        (cls.work_dir / "m.onnx.data").write_bytes(b"fake-external-weights")
        write_build_spec(
            {
                "trtc_build_spec": 1,
                "components": [
                    {
                        "onnx": "m.onnx",
                        "strongly_typed": True,
                        "profiles": [{"x": {"min": [1, 3], "opt": [2, 3], "max": [4, 3]}}],
                        "builder_config": {"flags": ["TF32"], "builder_optimization_level": 4},
                        "external_data": {"m.onnx.data": None},
                    }
                ],
            },
            cls.work_dir,
        )

        # Stand-in for the real trtc-build: gets the extracted job dir, checks
        # every job file arrived, engine = copy of the onnx, plus the
        # single-component manifest the real tool would write.
        stub = tmp / "fake-trtc-build"
        stub.write_text(
            textwrap.dedent(
                """\
                #!/bin/sh
                [ "$1" = "build" ] && shift   # the broker execs `<exe> build <dir> ...`
                input=$1; shift
                while [ $# -gt 0 ]; do
                  case $1 in
                    --out) out=$2; shift 2;;
                    *) shift;;
                  esac
                done
                for f in trtc_build_spec.json m.onnx m.onnx.data; do
                  [ -f "$input/$f" ] || { echo "missing $f in job dir"; exit 1; }
                done
                echo "stub build from $input"
                cp "$input/m.onnx" "$out/m.engine"
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
                "TRTC_TENSORRT_VERSION": "10.13.3.9",
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

    def test_info_reports_baked_tensorrt(self):
        info = builder_info(self.url, token="secret")
        self.assertEqual(info["tensorrt"], "10.13.3.9")
        self.assertIn("cache_dir", info)

    def test_rejects_malformed_jobs_at_submission(self):
        def post(body, note):
            request = urllib.request.Request(
                f"{self.url}/builds", data=body, method="POST",
                headers={"Authorization": "Bearer secret", "Content-Type": "application/x-tar"},
            )
            with self.assertRaises(urllib.error.HTTPError, msg=note) as raised:
                urllib.request.urlopen(request)
            self.assertEqual(raised.exception.code, 400, note)

        post(b"definitely not a tar", "garbage body")

        def tar_of(members):
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w") as archive:
                for name, payload in members.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))
            return buffer.getvalue()

        post(tar_of({"m.onnx": b"x"}), "no spec in tar")
        spec = json.dumps({"trtc_build_spec": 1, "components": [{"onnx": "m.onnx"}]}).encode()
        post(tar_of({"trtc_build_spec.json": spec}), "spec references onnx missing from tar")
        evil = json.dumps({"trtc_build_spec": 1, "components": [{"onnx": "../evil.onnx"}]}).encode()
        post(tar_of({"trtc_build_spec.json": evil, "m.onnx": b"x"}), "traversal onnx name")
        post(tar_of({"../evil": b"x", "trtc_build_spec.json": spec, "m.onnx": b"x"}), "traversal tar member")

    def test_openapi_contract_matches_routes(self):
        request = urllib.request.Request(
            f"{self.url}/openapi.json", headers={"Authorization": "Bearer secret"}
        )
        with urllib.request.urlopen(request) as response:
            spec = json.loads(response.read())
        self.assertEqual(spec["openapi"], "3.1.0")
        self.assertEqual(
            set(spec["paths"]),
            {"/builds", "/builds/{id}", "/builds/{id}/artifacts", "/info", "/openapi.json"},
        )
        # The broker-only test binary carries no TensorRT, so the option enums
        # are absent — but the option vocabulary itself is the contract.
        config = spec["components"]["schemas"]["BuilderConfig"]
        self.assertFalse(config["additionalProperties"])
        self.assertIn("flags", config["properties"])
        self.assertIn("memory_pool_limits", config["properties"])

    def test_unknown_job_is_404(self):
        request = urllib.request.Request(
            f"{self.url}/builds/eeeeeeeeeeee", headers={"Authorization": "Bearer secret"}
        )
        with self.assertRaises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(request)
        self.assertEqual(raised.exception.code, 404)

    def test_full_round_trip(self):
        spec = read_build_spec(self.work_dir)
        job_tar = pack_job_tar(single_component_spec(spec["components"][0]), self.work_dir)
        job_id = submit_build(self.url, job_tar, token="secret")
        job = wait_for_build(self.url, job_id, token="secret", poll_seconds=0.05, echo_log=False)
        self.assertEqual(job["state"], "succeeded", job.get("error"))
        self.assertEqual(job["result"], RESULT)
        # The job's spec is visible in its status, builder_config included.
        self.assertEqual(job["spec"]["components"][0]["builder_config"]["builder_optimization_level"], 4)

        dest = Path(self._tmp.name) / "out" / "m.engine"
        download_engine(self.url, job_id, dest, token="secret")
        self.assertEqual(dest.read_bytes(), b"fake-onnx-bytes")


if __name__ == "__main__":
    unittest.main()
