"""Dependency-free regression tests for manifest sample limits.

Run with ``python -m unittest discover -s tests -p test_validate_limits.py``.
Load the complete validation module directly so this pre-flight test does not
import the package's optional training or inference dependencies.
"""

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "voxcpm_validate_limits", ROOT / "src" / "voxcpm" / "training" / "validate.py"
)
validation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validation
SPEC.loader.exec_module(validation)


class _ReadLimitGuard:
    """Fail if validation asks the real file for a line after its budget."""

    def __init__(self, stream, max_lines):
        self.stream = stream
        self.max_lines = max_lines
        self.lines_read = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stream.close()

    def __iter__(self):
        return self

    def __next__(self):
        if self.lines_read >= self.max_lines:
            raise AssertionError("read beyond the requested sample limit")
        line = next(self.stream)
        self.lines_read += 1
        return line

    def readlines(self):
        return list(self)


class TestManifestSampleLimits(unittest.TestCase):
    def setUp(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)
        self.manifest = self.root / "train.jsonl"
        with wave.open(str(self.root / "audio.wav"), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16_000)
            audio.writeframes(b"\x00\x00" * 16_000)
        self.record = json.dumps({"text": "hello", "audio": "audio.wav"}) + "\n"

    def write_manifest(self, text):
        self.manifest.write_text(text, encoding="utf-8")

    def validate(self, **kwargs):
        return validation.validate_manifest(str(self.manifest), **kwargs)

    def test_leading_blank_lines_do_not_exhaust_limit(self):
        self.write_manifest("\n \t\n" + self.record + "not json\n")
        result = self.validate(max_samples=1)
        self.assertEqual(result.total_samples, 1)
        self.assertEqual(result.valid_samples, 1)
        self.assertTrue(result.is_valid)

    def test_interspersed_blanks_preserve_physical_error_line(self):
        self.write_manifest("\n" + self.record + " \nnot json\n" + self.record)
        result = self.validate(max_samples=2)
        self.assertEqual(result.total_samples, 2)
        self.assertEqual(result.valid_samples, 1)
        self.assertEqual(len(result.errors), 1)
        self.assertTrue(result.errors[0].startswith("Line 4: Invalid JSON"))

    def test_malformed_json_counts_toward_limit(self):
        self.write_manifest("not json\n" + self.record)
        result = self.validate(max_samples=1)
        self.assertEqual(result.total_samples, 1)
        self.assertEqual(result.valid_samples, 0)
        self.assertTrue(result.errors[0].startswith("Line 1: Invalid JSON"))

    def test_non_object_json_counts_toward_limit(self):
        self.write_manifest("[]\n" + self.record)
        result = self.validate(max_samples=1)
        self.assertEqual(result.total_samples, 1)
        self.assertEqual(result.errors, ["Line 1: Expected JSON object, got list"])

    def test_stops_reading_at_last_requested_sample(self):
        self.write_manifest("\n" + self.record + " \n" + self.record + "not json\n")
        reader = _ReadLimitGuard(self.manifest.open(encoding="utf-8"), max_lines=4)
        with patch.object(validation, "open", return_value=reader, create=True):
            result = self.validate(max_samples=2)
        self.assertEqual(reader.lines_read, 4)
        self.assertTrue(reader.stream.closed)
        self.assertEqual(result.total_samples, 2)
        self.assertEqual(result.valid_samples, 2)
        self.assertEqual(result.errors, [])

    def test_zero_and_negative_limits_validate_all_samples(self):
        self.write_manifest("\n" + self.record + " \n" + self.record + "not json\n")
        for limit in (0, -1, -10):
            with self.subTest(limit=limit):
                result = self.validate(max_samples=limit)
                self.assertEqual(result.total_samples, 3)
                self.assertEqual(result.valid_samples, 2)
                self.assertTrue(result.errors[0].startswith("Line 5: Invalid JSON"))

    def test_none_remains_an_unsupported_limit(self):
        self.write_manifest(self.record)
        with self.assertRaises(TypeError):
            self.validate(max_samples=None)

    def test_limit_larger_than_manifest_checks_every_sample(self):
        self.write_manifest("\n" + self.record + "\n" + self.record)
        result = self.validate(max_samples=10)
        self.assertEqual(result.total_samples, 2)
        self.assertEqual(result.valid_samples, 2)
        self.assertTrue(result.is_valid)

    def test_regular_manifest_behavior_is_unchanged(self):
        self.write_manifest(self.record * 3)
        for limit, count in ((0, 3), (1, 1), (2, 2), (3, 3), (10, 3)):
            with self.subTest(limit=limit):
                result = self.validate(max_samples=limit)
                self.assertEqual(result.total_samples, count)
                self.assertEqual(result.valid_samples, count)
                self.assertEqual(result.text_lengths, [5] * count)
                self.assertEqual(result.errors, [])
                self.assertEqual(result.warnings, [])

    def test_empty_and_all_blank_manifests_keep_existing_diagnostics(self):
        for limit in (0, 1, -1):
            for contents in ("", "\n \t\n"):
                with self.subTest(limit=limit, contents=contents):
                    self.write_manifest(contents)
                    result = self.validate(max_samples=limit)
                    self.assertEqual(result.total_samples, 0)
                    self.assertFalse(result.is_valid)
                    self.assertEqual(result.errors, ["Manifest file is empty"] if not contents else [])

    def test_verbose_progress_counts_samples_not_physical_lines(self):
        self.write_manifest(("\n" + self.record) * 100)
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            result = self.validate(max_samples=100, verbose=True)
        self.assertEqual(result.valid_samples, 100)
        self.assertEqual(output.getvalue(), "  Validated 100/100 samples...\n")


if __name__ == "__main__":
    unittest.main()
