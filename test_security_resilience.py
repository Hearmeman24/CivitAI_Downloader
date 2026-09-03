#!/usr/bin/env python3
"""Adversarial behavior tests for the downloader trust boundaries."""

import hashlib
import io
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import download_with_aria as dwa


def resource(payload=b"model payload", name="model.safetensors", file_id="30"):
    return dwa.ResolvedCivitAIResource(
        original="civitai:10@20+30",
        model_id="10",
        version_id="20",
        file_id=file_id,
        filename=name,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest().upper(),
        file_type="Model",
        file_format="SafeTensor",
    )


class PathConfinementTests(unittest.TestCase):
    def test_rejects_path_components_and_portable_dangerous_names(self):
        with tempfile.TemporaryDirectory() as td:
            downloader = dwa.CivitAIDownloader("token", td)
            for name in (
                "../victim.safetensors",
                "/tmp/victim.safetensors",
                "nested/model.safetensors",
                r"nested\model.safetensors",
                "CON.safetensors",
                ".",
                "model\nname.safetensors",
            ):
                with self.subTest(name=name), self.assertRaises(dwa.UnsafePathError):
                    downloader._safe_output_path(name)

    def test_force_cannot_delete_a_path_outside_output_directory(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            output = root / "output"
            output.mkdir()
            victim = root / "victim.safetensors"
            victim.write_bytes(b"important")
            downloader = dwa.CivitAIDownloader("token", output)

            with self.assertRaises(dwa.UnsafePathError):
                downloader._download_resource(
                    resource(name="../victim.safetensors"), force=True
                )

            self.assertEqual(victim.read_bytes(), b"important")

    def test_symlink_targets_are_rejected(self):
        if not hasattr(os, "symlink"):
            self.skipTest("symlinks are unavailable")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            outside = root / "outside.safetensors"
            outside.write_bytes(b"outside")
            output = root / "output"
            output.mkdir()
            (output / "model.safetensors").symlink_to(outside)
            downloader = dwa.CivitAIDownloader("token", output)

            with self.assertRaises(dwa.UnsafePathError):
                downloader._download_resource(resource(), force=True)

            self.assertEqual(outside.read_bytes(), b"outside")


class IntegrityAndSelectionTests(unittest.TestCase):
    def test_integrity_requires_exact_size_and_sha256(self):
        payload = b"verified bytes"
        expected = resource(payload=payload)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / expected.filename
            path.write_bytes(payload)
            downloader = dwa.CivitAIDownloader("token", td)
            verification = downloader._verify_file(path, expected)
            self.assertEqual(verification.size_bytes, len(payload))
            self.assertEqual(verification.sha256, expected.sha256)

            path.write_bytes(payload + b"corrupt")
            with self.assertRaises(dwa.IntegrityError):
                downloader._verify_file(path, expected)

    def test_missing_expected_file_never_falls_back_to_a_recent_file(self):
        payload = b"expected"
        expected = resource(payload=payload)
        with tempfile.TemporaryDirectory() as td:
            output = Path(td)
            unrelated = output / "unrelated.safetensors"
            unrelated.write_bytes(b"unrelated and recent")
            downloader = dwa.CivitAIDownloader("token", output)
            downloader._resolve_download_url = lambda _: "https://files.example/model"
            downloader._run_aria2c = lambda cmd, source: dwa.AriaResult(0, ())

            with self.assertRaisesRegex(dwa.DownloadError, "expected staging file"):
                downloader._download_resource(expected)

            self.assertEqual(unrelated.read_bytes(), b"unrelated and recent")

    def test_wrong_existing_file_is_preserved_and_not_treated_as_cached(self):
        payload = b"correct payload"
        expected = resource(payload=payload)
        with tempfile.TemporaryDirectory() as td:
            output = Path(td)
            original = output / expected.filename
            original.write_bytes(b"wrong but important")
            downloader = dwa.CivitAIDownloader("token", output)
            downloader._resolve_download_url = lambda _: "https://files.example/model"

            def fake_aria(cmd, source):
                out_name = next(
                    value.split("=", 1)[1]
                    for value in cmd
                    if value.startswith("--out=")
                )
                (output / out_name).write_bytes(payload)
                return dwa.AriaResult(0, ())

            downloader._run_aria2c = fake_aria
            outcome = downloader._download_resource(expected)

            self.assertEqual(original.read_bytes(), b"wrong but important")
            self.assertNotEqual(outcome.path, original)
            self.assertEqual(outcome.path.read_bytes(), payload)
            self.assertEqual(outcome.status, "downloaded")

    def test_hash_failure_never_promotes_staging_to_final_filename(self):
        expected = resource(payload=b"expected")
        with tempfile.TemporaryDirectory() as td:
            output = Path(td)
            downloader = dwa.CivitAIDownloader("token", output)
            downloader._resolve_download_url = lambda _: "https://files.example/model"

            def fake_aria(cmd, source):
                out_name = next(
                    value.split("=", 1)[1]
                    for value in cmd
                    if value.startswith("--out=")
                )
                (output / out_name).write_bytes(b"tampered")
                return dwa.AriaResult(0, ())

            downloader._run_aria2c = fake_aria
            with self.assertRaises(dwa.IntegrityError):
                downloader._download_resource(expected)

            self.assertFalse((output / expected.filename).exists())


class CredentialAndNetworkTests(unittest.TestCase):
    def test_download_url_is_sent_over_stdin_not_child_argv(self):
        captured = {}

        class Sink:
            def __init__(self):
                self.value = ""

            def write(self, value):
                self.value += value

            def close(self):
                pass

        class FakeProcess:
            def __init__(self, cmd, **kwargs):
                captured["cmd"] = cmd
                self.stdin = Sink()
                captured["stdin"] = self.stdin
                self.stdout = io.StringIO("")

            def wait(self):
                return 0

        secret_url = "https://files.example/model?Authorization=secret"
        with mock.patch.object(dwa.subprocess, "Popen", FakeProcess):
            result = dwa.CivitAIDownloader("token", tempfile.mkdtemp())._run_aria2c(
                ["aria2c", "--input-file=-"], secret_url
            )

        self.assertEqual(result.returncode, 0)
        self.assertNotIn(secret_url, captured["cmd"])
        self.assertEqual(captured["stdin"].value, secret_url + "\n")

    def test_interrupted_aria_process_is_terminated_and_reaped(self):
        class InterruptedOutput(io.StringIO):
            def __iter__(self):
                raise KeyboardInterrupt

        class FakeProcess:
            def __init__(self):
                self.stdin = io.StringIO()
                self.stdout = InterruptedOutput()
                self.running = True
                self.terminated = False
                self.waited = False

            def poll(self):
                return None if self.running else 1

            def terminate(self):
                self.terminated = True
                self.running = False

            def kill(self):
                self.running = False

            def wait(self, timeout=None):
                self.waited = True
                return 1

        process = FakeProcess()
        downloader = dwa.CivitAIDownloader("token", tempfile.mkdtemp())
        with (
            mock.patch.object(dwa.subprocess, "Popen", return_value=process),
            self.assertRaises(KeyboardInterrupt),
        ):
            downloader._run_aria2c(
                ["aria2c", "--input-file=-"], "https://files.example/model"
            )

        self.assertTrue(process.terminated)
        self.assertTrue(process.waited)

    def test_redirect_failure_does_not_return_token_bearing_original_url(self):
        class Response:
            status_code = 200

            def __init__(self):
                self.headers = {}

            def close(self):
                pass

        session = mock.Mock()
        session.get.return_value = Response()
        downloader = dwa.CivitAIDownloader(
            "secret", tempfile.mkdtemp(), session=session
        )

        with self.assertRaises(dwa.DownloadError):
            downloader._resolve_download_url(
                "https://civitai.com/api/download/models/20?token=secret"
            )

    def test_http_session_retries_rate_limits_and_transient_server_failures(self):
        downloader = dwa.CivitAIDownloader("token", tempfile.mkdtemp())
        retry = downloader.session.adapters["https://"].max_retries
        self.assertGreaterEqual(retry.total, 3)
        self.assertTrue({429, 500, 502, 503, 504}.issubset(retry.status_forcelist))
        self.assertTrue(retry.respect_retry_after_header)

        response = mock.Mock(headers={"Retry-After": "3600"})
        self.assertEqual(retry.get_retry_after(response), dwa.MAX_RETRY_AFTER_SECONDS)


class ConcurrencyAndCapacityTests(unittest.TestCase):
    def test_second_writer_for_same_filename_fails_immediately(self):
        with tempfile.TemporaryDirectory() as td:
            downloader = dwa.CivitAIDownloader("token", td)
            target = downloader._safe_output_path("model.safetensors")
            with (
                downloader._download_lock(target),
                self.assertRaises(dwa.ResourceBusyError),
                downloader._download_lock(target),
            ):
                pass

    def test_disk_preflight_reserves_space_before_transfer(self):
        with tempfile.TemporaryDirectory() as td:
            downloader = dwa.CivitAIDownloader("token", td)
            usage = mock.Mock(free=100)
            with (
                mock.patch.object(dwa.shutil, "disk_usage", return_value=usage),
                mock.patch.object(dwa, "DISK_RESERVE_BYTES", 64),
                self.assertRaisesRegex(dwa.DownloadError, "not enough disk space"),
            ):
                downloader._require_disk_space(64, "download")


class ArchiveSafetyTests(unittest.TestCase):
    def test_archive_member_paths_are_flattened_inside_output_directory(self):
        payload = b"safe tensor bytes"
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "output"
            output.mkdir()
            archive = output / "model.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("../../escape.safetensors", payload)

            downloader = dwa.CivitAIDownloader("token", output)
            artifacts = downloader._extract_safetensors(archive)

            self.assertEqual([path.name for path in artifacts], ["escape.safetensors"])
            self.assertEqual(artifacts[0].parent, output.resolve())
            self.assertFalse((Path(td) / "escape.safetensors").exists())

    def test_archive_expansion_limit_fails_before_extraction(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td)
            archive = output / "bomb.zip"
            with zipfile.ZipFile(
                archive, "w", compression=zipfile.ZIP_DEFLATED
            ) as bundle:
                bundle.writestr("large.safetensors", b"0" * 4096)

            downloader = dwa.CivitAIDownloader("token", output)
            with (
                mock.patch.object(dwa, "MAX_ARCHIVE_EXPANDED_BYTES", 1024),
                self.assertRaises(dwa.ArchiveSafetyError),
            ):
                downloader._extract_safetensors(archive)

    def test_archive_member_count_is_bounded(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td)
            archive = output / "many.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("one.safetensors", b"1")
                bundle.writestr("two.txt", b"2")

            downloader = dwa.CivitAIDownloader("token", output)
            with (
                mock.patch.object(dwa, "MAX_ARCHIVE_MEMBERS", 1),
                self.assertRaises(dwa.ArchiveSafetyError),
            ):
                downloader._extract_safetensors(archive)

    def test_archive_compression_ratio_is_bounded(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td)
            archive = output / "compressed.zip"
            with zipfile.ZipFile(
                archive, "w", compression=zipfile.ZIP_DEFLATED
            ) as bundle:
                bundle.writestr("model.safetensors", b"0" * 4096)

            downloader = dwa.CivitAIDownloader("token", output)
            with (
                mock.patch.object(dwa, "MAX_ARCHIVE_COMPRESSION_RATIO", 2),
                self.assertRaises(dwa.ArchiveSafetyError),
            ):
                downloader._extract_safetensors(archive)


if __name__ == "__main__":
    unittest.main()
