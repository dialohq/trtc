"""The builder: a deliberately dumb HTTP job broker on GPU hardware.

One job = one ONNX + build parameters in the query string = one engine back.
The server knows nothing about models, bundles, or plans — multi-component
compilation is N jobs, composed client-side. Each job runs `trtc build` in
this server's own interpreter, which is expected to be a correct, fixed
environment (trtc + the pinned tensorrt already installed, like a nix
derivation). There is no per-job dependency resolution: a plan pinning a
TensorRT this environment does not provide fails the job loudly, and you run a
builder image built for that version instead. Engine and timing caches persist
under TRTC_DATA_DIR, so a stopped-and-resumed instance stays warm.

API (bearer auth when TRTC_TOKEN is set):
  POST /builds?trt=<ver>[&name=][&dtype=][&workspace_gb=][&shape=NAME=MIN:OPT:MAX]*[&output_url=]
       body: raw ONNX bytes, or JSON {"input_url": ...} pointing at them
  GET  /builds/{id}[?log_offset=N]   status + log; "result" holds the build
       facts (single-component manifest) once succeeded
  GET  /builds/{id}/artifacts        the engine, as raw bytes
  GET  /info                         GPU facts, trtc version, cache location

Environment:
  TRTC_TOKEN         optional bearer token required on every request
  TRTC_DATA_DIR      jobs + caches root (default ~/.cache/trtc)
  TRTC_IDLE_TIMEOUT  seconds of inactivity before the server exits (0 = never)
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .plan import MANIFEST_FILE, query_gpu, read_json, write_json


class BuilderState:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.jobs_dir = data_dir / "jobs"
        self.cache_dir = data_dir / "cache"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.queue: queue.Queue[str] = queue.Queue()
        self.lock = threading.Lock()
        self.last_activity = time.monotonic()
        self.in_flight = 0

    def touch(self) -> None:
        self.last_activity = time.monotonic()

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def read_status(self, job_id: str) -> dict[str, Any] | None:
        status_path = self.job_dir(job_id) / "status.json"
        if not status_path.exists():
            return None
        return read_json(status_path)

    def write_status(self, job_id: str, **updates: Any) -> dict[str, Any]:
        with self.lock:
            status = self.read_status(job_id) or {"id": job_id}
            status.update(updates)
            write_json(self.job_dir(job_id) / "status.json", status)
            return status


_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def _safe_name(name: str) -> str:
    """A component name becomes filesystem paths on the builder, so reject
    anything that isn't a single innocuous path segment (no separators, no
    '..', no leading dot) to prevent traversal."""
    if not _SAFE_NAME.match(name) or ".." in name:
        raise ValueError(f"invalid name {name!r}: expected [A-Za-z0-9._-], no separators or '..'")
    return name


def parse_build_params(query: dict[str, list[str]]) -> dict[str, Any]:
    if not query.get("trt", [""])[0]:
        raise ValueError("missing required query parameter: trt (the tensorrt-cu12 pin)")
    return {
        "trt_version": query["trt"][0],
        "name": _safe_name(query.get("name", ["model"])[0]),
        "dtype": query.get("dtype", ["float32"])[0],
        "workspace_gb": float(query.get("workspace_gb", ["4"])[0]),
        "shapes": query.get("shape", []),
    }


def _build_command(state: BuilderState, onnx_path: Path, output_dir: Path, params: dict[str, Any]) -> list[str]:
    # Runs `trtc build` in this server's own interpreter. The builder is a
    # correct, fixed environment (trtc + the pinned tensorrt already installed,
    # like a nix derivation) — no per-job resolution. If the plan pins a
    # TensorRT this environment does not provide, build._check_trt_version
    # fails the job loudly; use a builder image built for that version instead.
    build_args = [
        "-m", "trtc.cli", "build", str(onnx_path),
        "--name", params["name"],
        "--dtype", params["dtype"],
        "--workspace-gb", str(params["workspace_gb"]),
        "--trt-version", params["trt_version"],
        "--out", str(output_dir),
        "--timing-cache", str(state.cache_dir / f"timing_trt{params['trt_version']}.bin"),
    ]
    for shape in params["shapes"]:
        build_args += ["--shape", shape]
    return [sys.executable, *build_args]


def _engine_path(state: BuilderState, status: dict[str, Any]) -> Path:
    return state.job_dir(status["id"]) / "engines" / f"{status['params']['name']}.engine"


def _run_job(state: BuilderState, job_id: str) -> None:
    job_dir = state.job_dir(job_id)
    output_dir = job_dir / "engines"
    log_path = job_dir / "job.log"
    status = state.read_status(job_id) or {}
    params = status["params"]
    with state.lock:
        state.in_flight += 1
    state.touch()
    state.write_status(job_id, state="running", started_at=time.time())
    try:
        onnx_path = job_dir / "input" / f"{params['name']}.onnx"
        input_url = status.get("input_url")
        if input_url:
            onnx_path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(input_url, timeout=600) as response:
                onnx_path.write_bytes(response.read())

        output_dir.mkdir(parents=True, exist_ok=True)
        command = _build_command(state, onnx_path, output_dir, params)

        environment = {**os.environ, "TRTC_CACHE_DIR": str(state.cache_dir)}
        with open(log_path, "ab") as log_handle:
            log_handle.write(f"$ {' '.join(command)}\n".encode())
            log_handle.flush()
            result = subprocess.run(command, stdout=log_handle, stderr=subprocess.STDOUT, env=environment)
        if result.returncode != 0:
            raise RuntimeError(f"build command exited with {result.returncode} (see job log)")

        output_url = status.get("output_url")
        if output_url:
            request = urllib.request.Request(
                output_url, data=_engine_path(state, status).read_bytes(), method="PUT"
            )
            request.add_header("Content-Type", "application/octet-stream")
            with urllib.request.urlopen(request, timeout=600) as response:
                if response.status not in (200, 201, 204):
                    raise RuntimeError(f"engine upload returned {response.status}")

        manifest_path = output_dir / MANIFEST_FILE
        build_result = read_json(manifest_path) if manifest_path.exists() else None
        state.write_status(job_id, state="succeeded", finished_at=time.time(), result=build_result)
    except Exception as error:  # noqa: BLE001 — job errors must land in status, not kill the worker
        with open(log_path, "ab") as log_handle:
            log_handle.write(f"ERROR: {error}\n".encode())
        state.write_status(job_id, state="failed", error=str(error), finished_at=time.time())
    finally:
        with state.lock:
            state.in_flight -= 1
        state.touch()


def _worker_loop(state: BuilderState) -> None:
    while True:
        job_id = state.queue.get()
        _run_job(state, job_id)


def _idle_watchdog(state: BuilderState, timeout: float) -> None:
    while True:
        time.sleep(30)
        # A job is dequeued before it runs, so also require no in-flight build
        # — otherwise a long build with no polling client is killed mid-flight.
        idle = state.queue.empty() and state.in_flight == 0
        if idle and time.monotonic() - state.last_activity > timeout:
            print(f"idle for {timeout:.0f}s, shutting down")
            os._exit(0)


def make_handler(state: BuilderState, token: str | None) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, code: int, payload: bytes, content_type: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_json(self, code: int, payload: dict[str, Any]) -> None:
            self._send(code, json.dumps(payload).encode())

        def _reject_unauthorized(self) -> bool:
            if token is None or self.headers.get("Authorization", "") == f"Bearer {token}":
                return False
            self._send_json(401, {"error": "unauthorized"})
            return True

        def do_GET(self) -> None:  # noqa: N802 — http.server API
            if self._reject_unauthorized():
                return
            state.touch()
            parsed = urllib.parse.urlparse(self.path)
            parts = [part for part in parsed.path.split("/") if part]

            if parts == ["info"]:
                jobs = sum(1 for path in state.jobs_dir.iterdir() if path.is_dir())
                self._send_json(
                    200,
                    {**query_gpu(), "trtc": _trtc_version(), "jobs": jobs, "cache_dir": str(state.cache_dir)},
                )
                return

            if len(parts) >= 2 and parts[0] == "builds":
                job_id = parts[1]
                status = state.read_status(job_id)
                if status is None:
                    self._send_json(404, {"error": f"unknown job {job_id}"})
                    return
                if len(parts) == 3 and parts[2] == "artifacts":
                    engine_path = _engine_path(state, status)
                    if status.get("state") != "succeeded" or not engine_path.exists():
                        self._send_json(409, {"error": f"job {job_id} has no engine (state={status.get('state')})"})
                        return
                    self._send(200, engine_path.read_bytes(), content_type="application/octet-stream")
                    return
                if len(parts) == 2:
                    query = urllib.parse.parse_qs(parsed.query)
                    log_offset = int(query.get("log_offset", ["0"])[0])
                    log_path = state.job_dir(job_id) / "job.log"
                    log_chunk = ""
                    new_offset = log_offset
                    if log_path.exists():
                        data = log_path.read_bytes()
                        log_chunk = data[log_offset:].decode(errors="replace")
                        new_offset = len(data)
                    self._send_json(200, {**status, "log": log_chunk, "log_offset": new_offset})
                    return

            self._send_json(404, {"error": f"no route for {self.path}"})

        def do_POST(self) -> None:  # noqa: N802 — http.server API
            if self._reject_unauthorized():
                return
            state.touch()
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path.rstrip("/") != "/builds":
                self._send_json(404, {"error": f"no route for {self.path}"})
                return

            query = urllib.parse.parse_qs(parsed.query)
            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length) if content_length else b""

            try:
                params = parse_build_params(query)
                status: dict[str, Any] = {
                    "state": "queued",
                    "created_at": time.time(),
                    "params": params,
                    "output_url": query.get("output_url", [None])[0],
                }
                if self.headers.get("Content-Type", "").startswith("application/json"):
                    job_request = json.loads(body)
                    if not job_request.get("input_url"):
                        raise ValueError("JSON submissions require input_url")
                    status["input_url"] = job_request["input_url"]
                elif body:
                    pass  # raw ONNX bytes; written below once the job dir exists
                else:
                    raise ValueError("empty body: send raw ONNX bytes or JSON with input_url")
            except (ValueError, json.JSONDecodeError) as error:
                self._send_json(400, {"error": str(error)})
                return

            job_id = uuid.uuid4().hex[:12]
            if "input_url" not in status:
                onnx_path = state.job_dir(job_id) / "input" / f"{params['name']}.onnx"
                onnx_path.parent.mkdir(parents=True, exist_ok=True)
                onnx_path.write_bytes(body)
            state.write_status(job_id, **status)
            state.queue.put(job_id)
            self._send_json(202, {"id": job_id, "state": "queued"})

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            sys.stderr.write(f"{self.address_string()} {format % args}\n")

    return Handler


def _trtc_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("trtc")
    except PackageNotFoundError:
        return "dev"


def serve(host: str = "0.0.0.0", port: int = 8080) -> None:
    data_dir = Path(os.getenv("TRTC_DATA_DIR", "~/.cache/trtc")).expanduser()
    state = BuilderState(data_dir)
    token = os.getenv("TRTC_TOKEN") or None

    threading.Thread(target=_worker_loop, args=(state,), daemon=True).start()
    idle_timeout = float(os.getenv("TRTC_IDLE_TIMEOUT", "0"))
    if idle_timeout > 0:
        threading.Thread(target=_idle_watchdog, args=(state, idle_timeout), daemon=True).start()

    server = ThreadingHTTPServer((host, port), make_handler(state, token))
    gpu = query_gpu()
    print(
        f"trtc builder on {host}:{port} — gpu={gpu['gpu_name']} cc={gpu['compute_capability']} "
        f"driver={gpu['driver_version']} data={data_dir}"
    )
    server.serve_forever()
