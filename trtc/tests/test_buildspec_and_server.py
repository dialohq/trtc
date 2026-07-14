from __future__ import annotations

import enum
import io
import json
import tarfile
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from trtc.buildspec import (
    BUILD_SPEC_FILE,
    BuilderConfig,
    BuildSpec,
    ComponentSpec,
    ShapeRange,
    assemble_manifest,
    extract_job_tar,
    pack_job_tar,
    parse_size,
    read_build_spec,
    resolve_build_target,
    single_component_spec,
    trt_versions_compatible,
    write_build_spec,
)
from trtc.client.remote import download_engine, submit_build, wait_for_build
from trtc.server.app import BuilderState, make_handler
from trtc.server.build import _engine_cache_key, apply_builder_config


def _spec() -> BuildSpec:
    return BuildSpec(
        components=[
            ComponentSpec(
                onnx="m.onnx",
                profiles=[{"x": ShapeRange(min=[1, 3], opt=[2, 3], max=[4, 3])}],
                builder_config=BuilderConfig(flags=["FP16"], memory_pool_limits={"WORKSPACE": "1G"}),
            )
        ],
    )


def _spec_dir(tmp_dir: Path) -> Path:
    work_dir = tmp_dir / "work"
    work_dir.mkdir()
    (work_dir / "m.onnx").write_bytes(b"not-really-onnx")
    write_build_spec(_spec(), work_dir)
    return work_dir


class BuildSpecTests(unittest.TestCase):
    def test_spec_roundtrips_through_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = _spec_dir(Path(tmp))
            spec = read_build_spec(work_dir)
        self.assertEqual(spec, _spec())
        component = spec.components[0]
        self.assertEqual(component.engine, "m.engine")  # derived from the onnx name
        self.assertEqual(component.profiles[0]["x"].opt, [2, 3])
        self.assertEqual(component.builder_config.memory_pool_limits, {"WORKSPACE": 1 << 30})

    def test_spec_rejects_unknown_and_missing_keys(self):
        data = _spec().to_dict()
        with self.assertRaises(ValueError):
            BuildSpec.from_dict({**data, "trtc_build_spec": 99})
        with self.assertRaises(ValueError):
            BuildSpec.from_dict({**data, "surprise": 1})
        with self.assertRaises(ValueError):  # typo'd IBuilderConfig attribute is caught at parse time
            component = {**data["components"][0], "builder_config": {"builder_optimisation_level": 4}}
            BuildSpec.from_dict({**data, "components": [component]})
        with self.assertRaises(ValueError):
            component = dict(data["components"][0])
            del component["onnx"]
            BuildSpec.from_dict({**data, "components": [component]})

    def test_builder_config_normalizes_names_and_sizes(self):
        config = BuilderConfig(
            flags=["fp16"],
            memory_pool_limits={"workspace": "4G", "DLA_LOCAL_DRAM": 512},
            profiling_verbosity="DETAILED",
        )
        self.assertEqual(config.flags, ["FP16"])
        self.assertEqual(config.memory_pool_limits, {"WORKSPACE": 4 << 30, "DLA_LOCAL_DRAM": 512})
        self.assertEqual(parse_size("1.5G"), int(1.5 * (1 << 30)))
        # to_dict keeps only deviations from TensorRT defaults.
        self.assertEqual(set(config.to_dict()), {"flags", "memory_pool_limits", "profiling_verbosity"})

    def test_shape_range_requires_equal_ranks(self):
        with self.assertRaises(ValueError):
            ShapeRange(min=[1, 3], opt=[2], max=[4, 3])

    def test_resolve_target_prefers_sibling_spec_and_falls_back_to_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = _spec_dir(Path(tmp))
            resolved_dir, spec = resolve_build_target(work_dir / "m.onnx")
            self.assertEqual(resolved_dir, work_dir.resolve())
            self.assertEqual(spec.components[0].builder_config.flags, ["FP16"])  # sibling spec wins

            bare = Path(tmp) / "bare"
            bare.mkdir()
            (bare / "solo.onnx").write_bytes(b"fake-onnx")
            resolved_dir, spec = resolve_build_target(bare / "solo.onnx")
            self.assertEqual(spec.components[0].name, "solo")
            self.assertEqual(spec.components[0].builder_config, BuilderConfig())  # TensorRT defaults
            self.assertEqual(sorted(p.name for p in bare.iterdir()), ["solo.onnx"])  # nothing written

    def test_engine_cache_key_tracks_identity(self):
        component = _spec().components[0]
        files = {"m.onnx": "abc"}
        key = _engine_cache_key(component, files, "10.13.3.9", "8.9")
        self.assertEqual(key, _engine_cache_key(component, dict(files), "10.13.3.9", "8.9"))
        self.assertNotEqual(key, _engine_cache_key(component, files, "10.13.3.9", "9.0"))
        self.assertNotEqual(key, _engine_cache_key(component, {"m.onnx": "def"}, "10.13.3.9", "8.9"))
        self.assertNotEqual(key, _engine_cache_key(component, {**files, "m.onnx.data": "eee"}, "10.13.3.9", "8.9"))
        reconfigured = ComponentSpec.from_dict(
            {**component.to_dict(), "builder_config": {"builder_optimization_level": 5}}
        )
        self.assertNotEqual(key, _engine_cache_key(reconfigured, files, "10.13.3.9", "8.9"))

    def test_assemble_manifest_merges_component_results(self):
        spec = _spec()
        build_facts = {"tensorrt_version": "10.13.3.9.post1", "compute_capability": "8.9"}
        result = {
            "components": [{"onnx": "m.onnx", "engine": "m.engine", "engine_sha256": "eee", "engine_size": 7}],
            "build": build_facts,
        }
        manifest = assemble_manifest(spec, [result])
        self.assertEqual(manifest["components"][0]["engine_sha256"], "eee")
        self.assertEqual(manifest["build"], build_facts)
        with self.assertRaises(ValueError):
            assemble_manifest(spec, [{**result, "components": [{**result["components"][0], "onnx": "other.onnx"}]}])


class ExternalDataTests(unittest.TestCase):
    """Large models (>2GB) keep weights in external data files next to the
    ONNX; those files belong to the component and travel with it."""

    def _spec_with_data(self, work_dir: Path) -> BuildSpec:
        (work_dir / "big.onnx").write_bytes(b"graph")
        (work_dir / "big.onnx.data").write_bytes(b"many gigabytes of weights")
        spec = BuildSpec(
            components=[ComponentSpec(onnx="big.onnx", external_data={"big.onnx.data": None})],
        )
        write_build_spec(spec, work_dir)
        return spec

    def test_external_data_roundtrips_through_spec_and_tar(self):
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            self._spec_with_data(work_dir)
            spec = read_build_spec(work_dir)
            self.assertEqual(spec.components[0].external_data, {"big.onnx.data": None})

            data = pack_job_tar(spec, work_dir)
            dest = work_dir / "job"
            extracted = extract_job_tar(data, dest)
            self.assertEqual((dest / "big.onnx.data").read_bytes(), b"many gigabytes of weights")
            self.assertEqual(extracted.components[0].external_data, {"big.onnx.data": None})

    def test_pack_requires_external_files_to_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            spec = self._spec_with_data(work_dir)
            (work_dir / "big.onnx.data").unlink()
            with self.assertRaises(FileNotFoundError):
                pack_job_tar(spec, work_dir)

    def test_extract_requires_declared_external_files_in_tar(self):
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            spec = self._spec_with_data(work_dir)
            data = pack_job_tar(spec, work_dir)
            # Rebuild the tar without the data file the spec declares.
            with tarfile.open(fileobj=io.BytesIO(data)) as archive:
                members = {
                    m.name: archive.extractfile(m).read() for m in archive.getmembers() if m.name != "big.onnx.data"
                }
            stripped = io.BytesIO()
            with tarfile.open(fileobj=stripped, mode="w") as archive:
                for name, payload in members.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))
            with self.assertRaises(ValueError):
                extract_job_tar(stripped.getvalue(), work_dir / "job")

    def test_extract_rejects_unsafe_external_names(self):
        bad = _spec().to_dict()
        bad["components"][0]["external_data"] = {"../evil.data": None}
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            for name, payload in {BUILD_SPEC_FILE: json.dumps(bad).encode(), "m.onnx": b"x"}.items():
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                extract_job_tar(buffer.getvalue(), Path(tmp) / "job")

    def test_default_spec_picks_up_sibling_data_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            (work_dir / "big.onnx").write_bytes(b"graph")
            (work_dir / "big.onnx.data").write_bytes(b"weights")
            _, spec = resolve_build_target(work_dir / "big.onnx")
            self.assertEqual(list(spec.components[0].external_data), ["big.onnx.data"])
            self.assertIsNotNone(spec.components[0].external_data["big.onnx.data"])  # sha filled in

            (work_dir / "big.onnx.data").unlink()
            _, spec = resolve_build_target(work_dir / "big.onnx")
            self.assertEqual(spec.components[0].external_data, {})

    def test_export_snapshot_spots_new_and_rewritten_files(self):
        from trtc.client.export import new_files

        self.assertEqual(new_files({"a": 1}, {"a": 1, "b": 2}), ["b"])
        self.assertEqual(new_files({"a": 1}, {"a": 9}), ["a"])  # rewritten counts
        self.assertEqual(new_files({"a": 1, "gone": 2}, {"a": 1}), [])


class JobTarTests(unittest.TestCase):
    """A builder job is one tar: the spec with its ONNX next to it."""

    def test_pack_extract_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = _spec_dir(Path(tmp))
            spec = read_build_spec(work_dir)
            data = pack_job_tar(single_component_spec(spec.components[0]), work_dir)

            dest = Path(tmp) / "job"
            extracted = extract_job_tar(data, dest)
            self.assertEqual(extracted.components[0].name, "m")
            self.assertEqual((dest / "m.onnx").read_bytes(), b"not-really-onnx")
            self.assertTrue((dest / BUILD_SPEC_FILE).exists())

    def test_pack_requires_single_component_and_existing_onnx(self):
        spec = _spec()
        two = BuildSpec(
            components=[spec.components[0], ComponentSpec(onnx="n.onnx")],
        )
        with self.assertRaises(ValueError):
            pack_job_tar(two, ".")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                pack_job_tar(spec, tmp)  # m.onnx missing

    def _tar_with(self, members: dict[str, bytes]) -> bytes:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            for name, payload in members.items():
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
        return buffer.getvalue()

    def test_extract_rejects_hostile_or_malformed_tars(self):
        spec_json = json.dumps(_spec().to_dict()).encode()
        cases = {
            "traversal member": {"../evil": b"x", BUILD_SPEC_FILE: spec_json, "m.onnx": b"x"},
            "nested member": {"a/b.onnx": b"x", BUILD_SPEC_FILE: spec_json},
            "no spec": {"m.onnx": b"x"},
            "spec without its onnx": {BUILD_SPEC_FILE: spec_json},
        }
        for label, members in cases.items():
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError, msg=label):
                    extract_job_tar(self._tar_with(members), Path(tmp) / "job")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                extract_job_tar(b"definitely not a tar", Path(tmp) / "job")

    def test_extract_rejects_unsafe_spec_file_names(self):
        bad = _spec().to_dict()
        bad["components"][0]["engine"] = "../../escape.engine"
        members = {BUILD_SPEC_FILE: json.dumps(bad).encode(), "m.onnx": b"x"}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                extract_job_tar(self._tar_with(members), Path(tmp) / "job")


class ApplyBuilderConfigTests(unittest.TestCase):
    """A BuilderConfig lands 1:1 on a (fake) tensorrt.IBuilderConfig: flags
    via set_flag, pools via set_memory_pool_limit, plain fields by attribute
    with enum members resolved by name."""

    class _FakeTrt:
        class BuilderFlag(enum.Enum):
            FP16 = 0
            TF32 = 1

        class MemoryPoolType(enum.Enum):
            WORKSPACE = 0
            DLA_LOCAL_DRAM = 1

        class QuantizationFlag(enum.Enum):
            CALIBRATE_BEFORE_FUSION = 0

        class TacticSource(enum.IntEnum):
            CUBLAS = 0
            CUDNN = 1

        class PreviewFeature(enum.Enum):
            PROFILE_SHARING_0806 = 0

    class _Verbosity(enum.Enum):
        LAYER_NAMES_ONLY = 0
        DETAILED = 1

    class _FakeConfig:
        def __init__(self, verbosity):
            self.builder_optimization_level = 3
            self.avg_timing_iterations = 1
            self.max_aux_streams = -1
            self.profiling_verbosity = verbosity
            self.set_flags = []
            self.pool_limits = {}
            self.quant_flags = []
            self.tactic_mask = None
            self.previews = {}

        def set_flag(self, flag):
            self.set_flags.append(flag)

        def set_memory_pool_limit(self, pool, limit):
            self.pool_limits[pool] = limit

        def set_quantization_flag(self, flag):
            self.quant_flags.append(flag)

        def set_tactic_sources(self, mask):
            self.tactic_mask = mask

        def set_preview_feature(self, feature, enabled):
            self.previews[feature] = enabled

    def _apply(self, builder_config: BuilderConfig):
        config = self._FakeConfig(self._Verbosity.LAYER_NAMES_ONLY)
        apply_builder_config(self._FakeTrt, config, builder_config)
        return config

    def test_applies_every_field_kind(self):
        trt = self._FakeTrt
        config = self._apply(
            BuilderConfig(
                flags=["FP16", "TF32"],
                memory_pool_limits={"WORKSPACE": 1 << 30},
                quantization_flags=["CALIBRATE_BEFORE_FUSION"],
                tactic_sources=["CUBLAS", "CUDNN"],
                preview_features={"PROFILE_SHARING_0806": True},
                builder_optimization_level=5,
                profiling_verbosity="DETAILED",
            )
        )
        self.assertEqual(config.set_flags, [trt.BuilderFlag.FP16, trt.BuilderFlag.TF32])
        self.assertEqual(config.pool_limits, {trt.MemoryPoolType.WORKSPACE: 1 << 30})
        self.assertEqual(config.quant_flags, [trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION])
        self.assertEqual(config.tactic_mask, 0b11)
        self.assertEqual(config.previews, {trt.PreviewFeature.PROFILE_SHARING_0806: True})
        self.assertEqual(config.builder_optimization_level, 5)
        self.assertEqual(config.profiling_verbosity, self._Verbosity.DETAILED)

    def test_unset_fields_leave_tensorrt_defaults_alone(self):
        config = self._apply(BuilderConfig())
        self.assertEqual(config.builder_optimization_level, 3)
        self.assertEqual(config.set_flags, [])
        self.assertEqual(config.pool_limits, {})
        self.assertIsNone(config.tactic_mask)

    def test_rejects_names_this_tensorrt_does_not_have(self):
        with self.assertRaises(ValueError):
            self._apply(BuilderConfig(flags=["NOT_A_FLAG"]))
        with self.assertRaises(ValueError):
            self._apply(BuilderConfig(memory_pool_limits={"NOT_A_POOL": 1}))
        with self.assertRaises(ValueError):
            self._apply(BuilderConfig(profiling_verbosity="NOT_A_MEMBER"))
        with self.assertRaises(ValueError):
            self._apply(BuilderConfig(runtime_platform="X"))  # attribute absent on this fake TRT


class VersionTests(unittest.TestCase):
    def test_tensorrt_version_comes_from_nearest_uv_lock(self):
        from trtc.buildspec import tensorrt_version_from_lock

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

    def test_repo_lock_resolves_tensorrt_pin(self):
        from trtc.buildspec import tensorrt_version_from_lock, trt_version_tuple

        repo_root = Path(__file__).resolve().parents[2]
        locked = tensorrt_version_from_lock(repo_root)
        # The repo lock must pin some TensorRT; the exact version moves with
        # the lock (the lock is the source of truth, not this test).
        self.assertIsNotNone(locked)
        self.assertGreaterEqual(len(trt_version_tuple(locked)), 2)

    def test_trt_version_compatibility_is_major_minor(self):
        self.assertTrue(trt_versions_compatible("10.13.3.9", "10.13.2.6"))
        self.assertFalse(trt_versions_compatible("10.13.3.9", "10.9.0.34"))
        self.assertFalse(trt_versions_compatible("10.13.3.9", "11.1.0.106"))

    def test_trt_pin_satisfied_ignores_post_suffix_but_not_minor(self):
        from trtc.buildspec import trt_pin_satisfied

        self.assertTrue(trt_pin_satisfied("10.13.3.9.post1", "10.13.3.9"))  # module drops .postN
        self.assertTrue(trt_pin_satisfied("10.13.3.9", "10.13.3.9"))
        self.assertFalse(trt_pin_satisfied("10.13.3.9", "10.1"))  # '10.1' must NOT prefix-match '10.13'
        self.assertFalse(trt_pin_satisfied("10.13.3.9", "10.13.2.6"))

    def test_nvidia_kernel_module_version_parses_proc(self):
        from trtc.buildspec import nvidia_kernel_module_version, query_gpu

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


class ServerRoundTripTests(unittest.TestCase):
    """Job tar up -> stubbed build -> engine bytes down."""

    RESULT = {
        "components": [{"onnx": "m.onnx", "engine": "m.engine", "engine_sha256": "abc", "engine_size": 9}],
        "build": {"tensorrt_version": "10.13.3.9.post1", "compute_capability": "8.9"},
    }

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp_path = Path(self._tmp.name)
        self.work_dir = _spec_dir(tmp_path)
        self.state = BuilderState(tmp_path / "data")
        # Stand-in for the real build subprocess: engine = copy of the onnx,
        # plus the single-component manifest trtc-server build would write.
        result_json = json.dumps(self.RESULT).replace('"', '\\"')
        self._build_command = patch(
            "trtc.server.app._build_command",
            lambda state, input_dir, output_dir: [
                "sh", "-c",
                f'cp {input_dir}/m.onnx {output_dir}/m.engine'
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
        from trtc.server.app import _run_job

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

    def test_rejects_malformed_job_at_submission(self):
        request = urllib.request.Request(
            f"{self.url}/builds", data=b"not a tar", method="POST",
            headers={"Authorization": "Bearer secret", "Content-Type": "application/x-tar"},
        )
        with self.assertRaises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(request)
        self.assertEqual(raised.exception.code, 400)

    def test_full_round_trip(self):
        spec = read_build_spec(self.work_dir)
        job_tar = pack_job_tar(spec, self.work_dir)
        job_id = submit_build(self.url, job_tar, token="secret")
        job = wait_for_build(self.url, job_id, token="secret", poll_seconds=0.05, echo_log=False)
        self.assertEqual(job["state"], "succeeded", job.get("error"))
        self.assertEqual(job["result"], self.RESULT)
        # The job's spec is visible in its status.
        self.assertEqual(job["spec"]["components"][0]["builder_config"]["flags"], ["FP16"])

        dest = Path(self._tmp.name) / "out" / "m.engine"
        download_engine(self.url, job_id, dest, token="secret")
        self.assertEqual(dest.read_bytes(), b"not-really-onnx")


if __name__ == "__main__":
    unittest.main()
