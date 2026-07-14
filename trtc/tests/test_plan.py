from __future__ import annotations

import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from trtc.plan import (
    BUILD_SPEC_FILE,
    assemble_manifest,
    engine_name,
    pack_job_tar,
    read_build_spec,
    resolve_build_target,
    single_component_spec,
    trt_pin_satisfied,
    trt_versions_compatible,
    write_build_spec,
)


def spec_dir(tmp_dir: Path) -> Path:
    work_dir = tmp_dir / "work"
    work_dir.mkdir()
    (work_dir / "m.onnx").write_bytes(b"not-really-onnx")
    write_build_spec(
        {
            "trtc_build_spec": 1,
            "components": [
                {
                    "onnx": "m.onnx",
                    "strongly_typed": True,
                    "profiles": [{"x": {"min": [1, 3], "opt": [2, 3], "max": [4, 3]}}],
                    "builder_config": {"flags": ["TF32"], "memory_pool_limits": {"WORKSPACE": "1G"}},
                }
            ],
        },
        work_dir,
    )
    return work_dir


class SpecTests(unittest.TestCase):
    def test_spec_roundtrip_and_version_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = spec_dir(Path(tmp))
            spec = read_build_spec(work_dir)
            self.assertEqual(engine_name(spec["components"][0]), "m.engine")
            write_build_spec({"trtc_build_spec": 99, "components": []}, work_dir)
            with self.assertRaises(ValueError):
                read_build_spec(work_dir)

    def test_resolve_target_prefers_sibling_spec_and_falls_back_to_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = spec_dir(Path(tmp))
            resolved, spec = resolve_build_target(work_dir / "m.onnx")
            self.assertEqual(resolved, work_dir.resolve())
            self.assertEqual(spec["components"][0]["builder_config"]["flags"], ["TF32"])  # sibling wins

            bare = Path(tmp) / "bare"
            bare.mkdir()
            (bare / "solo.onnx").write_bytes(b"fake-onnx")
            (bare / "solo.onnx.data").write_bytes(b"weights")
            _, spec = resolve_build_target(bare / "solo.onnx")
            self.assertEqual(spec["components"][0]["onnx"], "solo.onnx")
            self.assertIn("solo.onnx.data", spec["components"][0]["external_data"])  # >2GB convention
            self.assertEqual(sorted(p.name for p in bare.iterdir()), ["solo.onnx", "solo.onnx.data"])  # nothing written

    def test_pack_job_tar_contains_spec_and_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = spec_dir(Path(tmp))
            spec = read_build_spec(work_dir)
            spec["components"][0]["external_data"] = {"m.onnx.data": None}
            (work_dir / "m.onnx.data").write_bytes(b"many gigabytes")
            data = pack_job_tar(single_component_spec(spec["components"][0]), work_dir)
            with tarfile.open(fileobj=io.BytesIO(data)) as archive:
                names = sorted(archive.getnames())
                self.assertEqual(names, ["m.onnx", "m.onnx.data", BUILD_SPEC_FILE])
                packed = json.loads(archive.extractfile(BUILD_SPEC_FILE).read())
            self.assertEqual(packed["components"][0]["onnx"], "m.onnx")

            (work_dir / "m.onnx.data").unlink()
            with self.assertRaises(FileNotFoundError):
                pack_job_tar(single_component_spec(spec["components"][0]), work_dir)

    def test_assemble_manifest_merges_component_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = read_build_spec(spec_dir(Path(tmp)))
        build_facts = {"tensorrt_version": "10.13.3", "compute_capability": "8.9"}
        result = {
            "components": [{"onnx": "m.onnx", "engine": "m.engine", "engine_sha256": "eee", "engine_size": 7}],
            "build": build_facts,
        }
        manifest = assemble_manifest(spec, [result])
        self.assertEqual(manifest["components"][0]["engine_sha256"], "eee")
        self.assertEqual(manifest["build"], build_facts)
        with self.assertRaises(ValueError):
            assemble_manifest(spec, [{**result, "components": [{**result["components"][0], "onnx": "other.onnx"}]}])


class VersionTests(unittest.TestCase):
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

    def test_trt_pin_satisfied_is_major_minor_patch(self):
        self.assertTrue(trt_pin_satisfied("10.13.3.9.post1", "10.13.3"))  # wheel suffixes ignored
        self.assertTrue(trt_pin_satisfied("10.13.3.9", "10.13.3"))
        self.assertFalse(trt_pin_satisfied("10.13.3", "10.1"))  # '10.1' must NOT prefix-match '10.13'
        self.assertFalse(trt_pin_satisfied("10.13.3", "10.13.2"))

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
        facts = query_gpu()
        self.assertEqual(set(facts), {"gpu_name", "compute_capability", "driver_version"})


if __name__ == "__main__":
    unittest.main()
