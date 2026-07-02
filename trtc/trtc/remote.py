"""HTTP client for a trtc builder: one ONNX + parameters up, one engine down.

Multi-component models are composed here, client-side — the builder only ever
sees single-ONNX jobs. Stdlib only.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


class BuilderError(RuntimeError):
    pass


def _request(
    url: str,
    *,
    method: str = "GET",
    data: bytes | None = None,
    content_type: str | None = None,
    token: str | None = None,
    timeout: float = 60.0,
) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=data, method=method)
    if content_type:
        request.add_header("Content-Type", content_type)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def build_query(params: dict[str, Any], output_url: str | None = None) -> str:
    pairs: list[tuple[str, str]] = [
        ("trt", params["trt_version"]),
        ("name", params["name"]),
        ("dtype", params["dtype"]),
        ("workspace_gb", str(params["workspace_gb"])),
    ]
    pairs += [("shape", shape) for shape in params.get("shapes", [])]
    if output_url:
        pairs.append(("output_url", output_url))
    return urllib.parse.urlencode(pairs)


def builder_info(builder_url: str, *, token: str | None = None) -> dict[str, Any]:
    status, body = _request(f"{builder_url.rstrip('/')}/info", token=token)
    if status != 200:
        raise BuilderError(f"Builder /info returned {status}: {body[:500].decode(errors='replace')}")
    return json.loads(body)


def submit_build(
    builder_url: str,
    onnx: str | Path | None = None,
    *,
    params: dict[str, Any],
    input_url: str | None = None,
    output_url: str | None = None,
    token: str | None = None,
) -> str:
    """Submit one ONNX for building; returns the job id. Either upload the
    bytes or point the builder at a presigned GET URL for them."""
    if (onnx is None) == (input_url is None):
        raise ValueError("Pass exactly one of onnx or input_url")
    url = f"{builder_url.rstrip('/')}/builds?{build_query(params, output_url)}"
    if onnx is not None:
        status, body = _request(
            url,
            method="POST",
            data=Path(onnx).read_bytes(),
            content_type="application/octet-stream",
            token=token,
            timeout=600.0,
        )
    else:
        status, body = _request(
            url,
            method="POST",
            data=json.dumps({"input_url": input_url}).encode(),
            content_type="application/json",
            token=token,
        )
    if status not in (200, 201, 202):
        raise BuilderError(f"Build submission failed ({status}): {body[:500].decode(errors='replace')}")
    return json.loads(body)["id"]


def wait_for_build(
    builder_url: str,
    job_id: str,
    *,
    token: str | None = None,
    poll_seconds: float = 5.0,
    echo_log: bool = True,
    max_consecutive_failures: int = 10,
) -> dict[str, Any]:
    base = builder_url.rstrip("/")
    log_offset = 0
    failures = 0
    while True:
        # Transient network errors must not abandon a running remote build.
        try:
            status, body = _request(f"{base}/builds/{job_id}?log_offset={log_offset}", token=token)
        except OSError as error:
            failures += 1
            if failures >= max_consecutive_failures:
                raise BuilderError(f"Lost contact with builder after {failures} attempts: {error}") from error
            time.sleep(poll_seconds)
            continue
        if status != 200:
            failures += 1
            if failures >= max_consecutive_failures:
                raise BuilderError(f"Status poll failed ({status}): {body[:500].decode(errors='replace')}")
            time.sleep(poll_seconds)
            continue
        failures = 0
        job = json.loads(body)
        log_chunk = job.get("log", "")
        if echo_log and log_chunk:
            print(log_chunk, end="", flush=True)
        log_offset = job.get("log_offset", log_offset)
        if job["state"] in ("succeeded", "failed"):
            return job
        time.sleep(poll_seconds)


def download_engine(
    builder_url: str,
    job_id: str,
    dest: str | Path,
    *,
    token: str | None = None,
) -> None:
    base = builder_url.rstrip("/")
    last_error: Exception | None = None
    for _ in range(3):
        try:
            status, body = _request(f"{base}/builds/{job_id}/artifacts", token=token, timeout=600.0)
        except OSError as error:
            last_error = error
            time.sleep(5)
            continue
        if status != 200:
            raise BuilderError(f"Engine download failed ({status}): {body[:500].decode(errors='replace')}")
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(body)
        return
    raise BuilderError(f"Engine download failed after retries: {last_error}")
