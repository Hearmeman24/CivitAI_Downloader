#!/usr/bin/env python3
"""Behavior tests for cached, resumable, stale, and forced transfer state."""

import hashlib
import os
import tempfile
import unittest
from pathlib import Path

import download_with_aria as dwa

PAYLOAD = b"verified model payload"
SHA256 = hashlib.sha256(PAYLOAD).hexdigest().upper()


def resource():
    return dwa.ResolvedCivitAIResource(
        original="20",
        model_id="10",
        version_id="20",
        file_id="30",
        filename="model.safetensors",
        size_bytes=len(PAYLOAD),
        sha256=SHA256,
        file_type="Model",
        file_format="SafeTensor",
    )


def staging_paths(output: Path):
    key = hashlib.sha256(b"model.safetensors").hexdigest()[:12]
    staging = output / f".civitai-30-{key}.part"
    return staging, Path(f"{staging}{dwa.ARIA2_EXT}")


class ResumeTests(unittest.TestCase):
    def run_case(self, setup, force=False):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td)
            staging, control = staging_paths(output)
            setup(output, staging, control)
            downloader = dwa.CivitAIDownloader("token", output)
            downloader._resolve_download_url = lambda _: "https://files.example/signed"
            captured = {"calls": 0}

            def fake_aria(cmd, source):
                captured["calls"] += 1
                captured["control_at_invoke"] = control.exists()
                captured["staging_size_at_invoke"] = (
                    staging.stat().st_size if staging.exists() else None
                )
                out_name = next(
                    item.split("=", 1)[1] for item in cmd if item.startswith("--out=")
                )
                self.assertEqual(out_name, staging.name)
                staging.write_bytes(PAYLOAD)
                if os.path.lexists(control):
                    control.unlink()
                return dwa.AriaResult(0, ())

            downloader._run_aria2c = fake_aria
            outcome = downloader._download_resource(resource(), force=force)
            captured["outcome"] = outcome
            captured["bytes"] = outcome.path.read_bytes()
            return captured

    def test_partial_with_control_resumes_same_staging_file(self):
        def setup(output, staging, control):
            staging.write_bytes(PAYLOAD[:5])
            control.write_bytes(b"aria state")

        captured = self.run_case(setup)
        self.assertEqual(captured["calls"], 1)
        self.assertTrue(captured["control_at_invoke"])
        self.assertEqual(captured["staging_size_at_invoke"], 5)
        self.assertEqual(captured["outcome"].status, "resumed")
        self.assertEqual(captured["bytes"], PAYLOAD)

    def test_orphaned_partial_is_discarded_before_fresh_download(self):
        def setup(output, staging, control):
            staging.write_bytes(b"bad partial")

        captured = self.run_case(setup)
        self.assertIsNone(captured["staging_size_at_invoke"])
        self.assertEqual(captured["outcome"].status, "downloaded")

    def test_verified_existing_target_is_cached(self):
        def setup(output, staging, control):
            (output / "model.safetensors").write_bytes(PAYLOAD)

        captured = self.run_case(setup)
        self.assertEqual(captured["calls"], 0)
        self.assertEqual(captured["outcome"].status, "cached")

    def test_force_removes_target_partial_and_control_before_download(self):
        def setup(output, staging, control):
            (output / "model.safetensors").write_bytes(PAYLOAD)
            staging.write_bytes(b"partial")
            control.write_bytes(b"aria state")

        captured = self.run_case(setup, force=True)
        self.assertFalse(captured["control_at_invoke"])
        self.assertIsNone(captured["staging_size_at_invoke"])
        self.assertEqual(captured["outcome"].status, "downloaded")

    def test_orphaned_control_is_removed_before_fresh_download(self):
        def setup(output, staging, control):
            control.write_bytes(b"orphaned aria state")

        captured = self.run_case(setup)
        self.assertFalse(captured["control_at_invoke"])
        self.assertEqual(captured["outcome"].status, "downloaded")


if __name__ == "__main__":
    unittest.main()
