from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from trtc.spec import Axis, Component, T, load_entry, parse_options


def _styletts2_decoder_component() -> Component:
    batch = Axis("batch", 1, 8, 16)
    asr_frames = Axis("asr_frames", 128, 256, 1024)
    f0_frames = asr_frames.affine(scale=2, name="f0_frames")
    har_frames = asr_frames.affine(scale=120, offset=1, name="har_frames")
    return Component(
        name="decoder",
        module=lambda: None,
        inputs={
            "asr": T([batch, 512, asr_frames]),
            "f0": T([batch, f0_frames]),
            "noise": T([batch, f0_frames]),
            "style": T([batch, 128]),
            "har": T([batch, 22, har_frames]),
        },
        outputs=["audio"],
        output_axes={"audio": {0: "batch", 2: "audio_samples"}},
        file_stem="decoder_dynamic",
    )


class AxisTests(unittest.TestCase):
    def test_axis_validates_ordering(self):
        with self.assertRaises(ValueError):
            Axis("bad", 8, 4, 16)
        with self.assertRaises(ValueError):
            Axis("bad", 0, 1, 2)

    def test_affine_values(self):
        asr = Axis("asr_frames", 128, 256, 1024)
        har = asr.affine(scale=120, offset=1, name="har_frames")
        self.assertEqual(har.value("min"), 15361)
        self.assertEqual(har.value("opt"), 30721)
        self.assertEqual(har.value("max"), 122881)


class ComponentTests(unittest.TestCase):
    def test_profiles_match_legacy_decoder_profile_shapes(self):
        # Exact output of the deleted decoder_dynamic_profile_shapes(min_batch=1,
        # opt_batch=8, max_batch=16, min_asr_frames=128, opt_asr_frames=256,
        # max_asr_frames=1024) — the bundle declaration must reproduce it.
        expected = {
            "asr": {"min": [1, 512, 128], "opt": [8, 512, 256], "max": [16, 512, 1024]},
            "f0": {"min": [1, 256], "opt": [8, 512], "max": [16, 2048]},
            "noise": {"min": [1, 256], "opt": [8, 512], "max": [16, 2048]},
            "style": {"min": [1, 128], "opt": [8, 128], "max": [16, 128]},
            "har": {"min": [1, 22, 15361], "opt": [8, 22, 30721], "max": [16, 22, 122881]},
        }
        self.assertEqual(_styletts2_decoder_component().profiles(), expected)

    def test_dynamic_axes_cover_inputs_and_outputs(self):
        axes = _styletts2_decoder_component().dynamic_axes()
        self.assertEqual(axes["asr"], {0: "batch", 2: "asr_frames"})
        self.assertEqual(axes["f0"], {0: "batch", 1: "f0_frames"})
        self.assertEqual(axes["style"], {0: "batch"})
        self.assertEqual(axes["audio"], {0: "batch", 2: "audio_samples"})

    def test_file_names_follow_stem(self):
        component = _styletts2_decoder_component()
        self.assertEqual(component.onnx_name, "decoder_dynamic.onnx")
        self.assertEqual(component.engine_name, "decoder_dynamic.engine")


class EntryTests(unittest.TestCase):
    def test_load_entry_from_file(self):
        source = textwrap.dedent(
            """
            from trtc.spec import Axis, Bundle, Component, T

            def bundle(weights, *, opt_batch=4):
                axis = Axis("batch", 1, opt_batch, 8)
                component = Component(
                    name="m", module=lambda: None, inputs={"x": T([axis, 3])}, outputs=["y"],
                )
                return Bundle(name="test", components=[component], meta={"weights": weights})
            """
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            entry_path = Path(tmp_dir) / "entry.py"
            entry_path.write_text(source)
            factory = load_entry(str(entry_path))
            bundle = factory("w.pth", opt_batch=2)
            self.assertEqual(bundle.meta["weights"], "w.pth")
            self.assertEqual(bundle.components[0].inputs["x"].shape("opt"), (2, 3))

    def test_dotted_module_with_py_segment_is_not_treated_as_file(self):
        # A module path containing '.py...' (e.g. '.pyworld') must route to the
        # import branch, not be misread as a file path.
        with self.assertRaises(ModuleNotFoundError):
            load_entry("nonexistent_pkg.pyworld_utils.bundle:bundle")

    def test_load_entry_from_package_file_supports_relative_imports(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pkg = Path(tmp_dir) / "mypkg"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")
            (pkg / "_helper.py").write_text("VALUE = 42\n")
            (pkg / "bundle.py").write_text(
                "from trtc.spec import Axis, Bundle, Component, T\n"
                "from ._helper import VALUE\n"
                "def bundle():\n"
                "    axis = Axis('batch', 1, VALUE, 64)\n"
                "    c = Component(name='m', module=lambda: None, inputs={'x': T([axis])}, outputs=['y'])\n"
                "    return Bundle(name='t', components=[c])\n"
            )
            factory = load_entry(str(pkg / "bundle.py"))
            self.assertEqual(factory().components[0].inputs["x"].shape("opt"), (42,))

    def test_parse_options(self):
        options = parse_options(["a=1", "b=1.5", "c=true", "d=x,y", "e=10,20", "f=text"])
        self.assertEqual(
            options,
            {"a": 1, "b": 1.5, "c": True, "d": ["x", "y"], "e": [10, 20], "f": "text"},
        )


if __name__ == "__main__":
    unittest.main()
