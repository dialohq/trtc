"""Provision a trtc builder on vast.ai and wait until it answers.

Shells out to the `vastai` CLI (install via the `launch` extra: `trtc[launch]`)
and polls the builder's /info. Stdlib only; no SDK dependency.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
import urllib.request
from typing import Any

DEFAULT_REGISTRY = "ghcr.io/dialohq/trtc-builder"
BUILDER_PORT = 8080


def log(message: str) -> None:
    print(f"[launch] {message}", file=sys.stderr, flush=True)


def _vastai(args: list[str], *, timeout: float = 120.0) -> subprocess.CompletedProcess:
    base = shlex.split(os.getenv("TRTC_VASTAI_CMD", "vastai"))
    key = os.getenv("VAST_API_KEY")
    # --api-key must precede any trailing `--args` (which consumes the rest of
    # the line), so slot it right after the verb+noun subcommand.
    key_flag = ["--api-key", key] if key else []
    cmd = base + args[:2] + key_flag + args[2:]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _vastai_json(args: list[str]) -> Any:
    result = _vastai(args)
    if result.returncode != 0:
        raise RuntimeError(f"vastai {' '.join(args)} failed: {result.stderr.strip() or result.stdout.strip()}")
    return json.loads(result.stdout)


def search_offers(query: str, limit: int) -> list[int]:
    offers = _vastai_json(["search", "offers", query, "-o", "dph", "--raw"])
    return [int(o["id"]) for o in offers[:limit]]


def create_instance(offer: int, *, image: str, disk: int, env: str, login: str | None) -> int | None:
    args = ["create", "instance", str(offer), "--image", image, "--disk", str(disk), "--env", env,
            "--cancel-unavail", "--raw"]
    if login:
        args += ["--login", login]
    # Args-launch mode: run the image's server directly. Without --args, vast
    # injects its own ssh/jupyter entrypoint and ignores the image CMD, so the
    # builder never starts. Keep --args last (it consumes the rest of the line).
    args += ["--entrypoint", "/bin/trtc-server", "--args", "serve",
             "--host", "0.0.0.0", "--port", str(BUILDER_PORT)]
    result = _vastai(args)
    if result.returncode != 0:
        log(f"offer {offer} unavailable: {result.stderr.strip() or result.stdout.strip()}")
        return None
    try:
        contract = json.loads(result.stdout).get("new_contract")
    except json.JSONDecodeError:
        return None
    return int(contract) if contract else None


def destroy_instance(instance: int) -> None:
    _vastai(["destroy", "instance", str(instance)])


def wait_for_endpoint(instance: int, *, port: int, attempts: int = 120, delay: float = 10.0) -> tuple[str, str]:
    for _ in range(attempts):
        time.sleep(delay)
        try:
            raw = _vastai_json(["show", "instance", str(instance), "--raw"])
        except (RuntimeError, json.JSONDecodeError):
            continue
        if raw.get("actual_status") == "running":
            ip = raw.get("public_ipaddr")
            ports = raw.get("ports") or {}
            mapping = ports.get(f"{port}/tcp") or []
            host_port = mapping[0].get("HostPort") if mapping else None
            if ip and host_port:
                return ip, host_port
        log(f"instance {instance} is {raw.get('actual_status', 'unknown')}...")
    raise RuntimeError(f"instance {instance} did not expose port {port} in time")


def wait_for_builder(url: str, *, token: str | None, attempts: int = 60, delay: float = 5.0) -> dict[str, Any] | None:
    request = urllib.request.Request(f"{url}/info")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return json.loads(response.read())
        except OSError:
            time.sleep(delay)
    return None


def launch(
    *,
    image: str,
    gpu: str,
    disk: int,
    token: str | None,
    idle_timeout: int | None,
    login: str | None,
    offers: int,
    query: str | None,
) -> str:
    # Require genuinely fast internet: the builder image is large (TensorRT +
    # CUDA), and slow/flaky hosts get stuck retrying the pull.
    query = query or (
        f"gpu_name={gpu} rentable=true verified=true num_gpus=1 reliability>0.98 "
        f"inet_down>2000 inet_up>2000 disk_space>{disk}"
    )
    env = f"-p {BUILDER_PORT}:{BUILDER_PORT}"
    if token:
        env += f" -e TRTC_TOKEN={token}"
    if idle_timeout:
        env += f" -e TRTC_IDLE_TIMEOUT={idle_timeout}"

    log(f"image: {image}")
    log(f"searching offers: {query}")
    offer_ids = search_offers(query, offers)
    if not offer_ids:
        raise RuntimeError("no matching vast.ai offers")

    instance = None
    for offer in offer_ids:
        log(f"trying offer {offer}")
        instance = create_instance(offer, image=image, disk=disk, env=env, login=login)
        if instance:
            log(f"created instance {instance} (destroy: vastai destroy instance {instance})")
            break
    if not instance:
        raise RuntimeError("could not rent any offer")

    ip, host_port = wait_for_endpoint(instance, port=BUILDER_PORT)
    url = f"http://{ip}:{host_port}"
    log(f"waiting for the builder to answer at {url}/info")
    info = wait_for_builder(url, token=token)
    if info is None:
        raise RuntimeError(f"builder at {url} did not answer; instance {instance} left running for inspection")
    log(f"builder ready: {json.dumps(info)}")
    log(f"destroy with: vastai destroy instance {instance}")
    return url
