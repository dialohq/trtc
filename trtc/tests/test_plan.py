from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from trtc.plan import (
    PLAN_FILE,
    assemble_manifest,
    build_params,
    plan_for_onnx,
    read_plan,
    shape_specs,
    trt_versions_compatible,
    write_json,
)

def _fake_plan_dir(tmp_dir: Path) -> Path:
    work_dir = tmp_dir / "work"
    work_dir.mkdir()
    (work_dir / "m.onnx").write_bytes(b"not-really-onnx")
    write_json(
        work_dir / PLAN_FILE,
        {
            "trtc_plan": 1,
            "bundle": "test",
            "tensorrt_version": "10.13.3.9.post1",
            "engine_dir_hint": None,
            "meta": {},
            "provenance": {},
            "components": [
                {
                    "name": "m",
                    "onnx": "m.onnx",
                    "engine": "m.engine",
                    "dtype": "float16",
                    "workspace_bytes": 1024,
                    "strongly_typed": True,
                    "profiles": {"x": {"min": [1, 3], "opt": [2, 3], "max": [4, 3]}},
                    "onnx_sha256": "abc",
                    "meta": {},
                }
            ],
        },
    )
    return work_dir


class PlanTests(unittest.TestCase):
    def test_read_plan_roundtrip_and_version_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = _fake_plan_dir(Path(tmp))
            plan = read_plan(work_dir)
            self.assertEqual(plan["components"][0]["engine"], "m.engine")

    def test_tensorrt_version_comes_from_nearest_uv_lock(self):
        from trtc.plan import tensorrt_version_from_lock

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "uv.lock").write_text(
                '[[package]]\nname = "torch"\nversion = "2.10.0"\n\n'
                '[[package]]\nname = "tensorrt-cu12"\nversion = "10.13.3.9.post1"\n'
            )
            nested = root / "a" / "b"
            nested.mkdir(parents=True)
            self.assertEqual(tensorrt_version_from_lock(nested), "10.13.3.9.post1")

            (root / "a" / "uv.lock").write_text('[[package]]\nname = "torch"\nversion = "2.10.0"\n')
            self.assertIsNone(tensorrt_version_from_lock(nested))  # nearest lock wins, even without a pin

    def test_trt_version_compatibility_is_major_minor(self):
        self.assertTrue(trt_versions_compatible("10.13.3.9", "10.13.2.6"))
        self.assertFalse(trt_versions_compatible("10.13.3.9", "10.9.0.34"))
        self.assertFalse(trt_versions_compatible("10.13.3.9", "11.1.0.106"))

    def test_nvidia_kernel_module_version_parses_proc(self):
        from trtc.plan import nvidia_kernel_module_version, query_gpu

        proc = (
            "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  590.48.01  Release Build"
            "  (dvs-builder@U16-I3-D08-2-2)  Mon Nov 24 04:14:44 UTC 2025\n"
            "GCC version:  gcc version 13.3.0\n"
        )
        self.assertEqual(nvidia_kernel_module_version(proc), "590.48.01")
        self.assertIsNone(nvidia_kernel_module_version("no driver here"))
        # Off-GPU boxes degrade to None without raising.
        facts = query_gpu()
        self.assertEqual(set(facts), {"gpu_name", "compute_capability", "driver_version"})

    def test_trt_pin_satisfied_ignores_post_suffix_but_not_minor(self):
        from trtc.plan import trt_pin_satisfied

        self.assertTrue(trt_pin_satisfied("10.13.3.9.post1", "10.13.3.9"))  # module drops .postN
        self.assertTrue(trt_pin_satisfied("10.13.3.9", "10.13.3.9"))
        self.assertFalse(trt_pin_satisfied("10.13.3.9", "10.1"))  # '10.1' must NOT prefix-match '10.13'
        self.assertFalse(trt_pin_satisfied("10.13.3.9", "10.13.2.6"))


class BareOnnxTests(unittest.TestCase):
    """A bare .onnx needs no plan file, and shapes round-trip through the
    CLI/wire string encoding."""

    def test_parse_shape_profile_roundtrip(self):
        from trtc.cli import parse_shape_profile

        name, profile = parse_shape_profile("asr=1x512x128:8x512x256:16x512x1024")
        self.assertEqual(name, "asr")
        self.assertEqual(profile, {"min": [1, 512, 128], "opt": [8, 512, 256], "max": [16, 512, 1024]})
        self.assertEqual(shape_specs({name: profile}), ["asr=1x512x128:8x512x256:16x512x1024"])
        with self.assertRaises(SystemExit):
            parse_shape_profile("asr=1x2:3x4")
        with self.assertRaises(SystemExit):
            parse_shape_profile("noshapes")

    def test_plan_for_onnx_and_job_params(self):
        with tempfile.TemporaryDirectory() as tmp:
            onnx = Path(tmp) / "m.onnx"
            onnx.write_bytes(b"fake-onnx")
            plan = plan_for_onnx(
                onnx,
                tensorrt_version="10.13.3.9.post1",
                dtype="float16",
                profiles={"x": {"min": [1, 3], "opt": [2, 3], "max": [4, 3]}},
            )
            self.assertEqual(plan["components"][0]["engine"], "m.engine")
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["m.onnx"])  # nothing written

            params = build_params(plan["components"][0], plan["tensorrt_version"])
            self.assertEqual(params["name"], "m")
            self.assertEqual(params["shapes"], ["x=1x3:2x3:4x3"])

    def test_assemble_manifest_merges_component_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = _fake_plan_dir(Path(tmp))
            plan = read_plan(work_dir)
        build_facts = {"tensorrt_version": "10.13.3.9.post1", "compute_capability": "8.9"}
        result = {
            "components": [{"name": "m", "engine": "m.engine", "engine_sha256": "eee", "engine_size": 7}],
            "build": build_facts,
        }
        manifest = assemble_manifest(plan, [result])
        self.assertEqual(manifest["components"][0]["engine_sha256"], "eee")
        self.assertEqual(manifest["build"], build_facts)
        with self.assertRaises(ValueError):
            assemble_manifest(plan, [{**result, "components": [{**result["components"][0], "name": "other"}]}])


if __name__ == "__main__":
    unittest.main()
