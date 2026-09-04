#!/usr/bin/env python3
"""The default CLI logging contract is concise and machine-readable enough."""

import hashlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import download_with_aria as dwa


class StructuredLoggingTests(unittest.TestCase):
    def test_legacy_long_model_and_short_token_flags_still_parse(self):
        args = dwa.build_parser().parse_args(["--model", "20", "-t", "token"])
        self.assertEqual(args.identifier, "20")
        self.assertEqual(args.token, "token")

    def test_success_is_exactly_two_structured_lines(self):
        stream = io.StringIO()
        logger = dwa.StructuredLogger(stream=stream)
        payload = b"model"
        resolved = dwa.ResolvedCivitAIResource(
            original="20",
            model_id="10",
            version_id="20",
            file_id="30",
            filename="model.safetensors",
            size_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest().upper(),
            file_type="Model",
            file_format="SafeTensor",
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / resolved.filename
            path.write_bytes(payload)
            downloader = dwa.CivitAIDownloader("token", td, logger=logger)
            downloader.resolve_identifier = lambda _: resolved
            downloader._download_resource = lambda *args, **kwargs: dwa.DownloadOutcome(
                path=path,
                artifacts=(path,),
                status="downloaded",
                size_bytes=len(payload),
                sha256=resolved.sha256,
            )

            outcome = downloader.download("20")

        lines = stream.getvalue().splitlines()
        self.assertEqual(len(lines), 2, stream.getvalue())
        self.assertRegex(lines[0], r"^INFO resolve ")
        self.assertIn("model=10", lines[0])
        self.assertIn("version=20", lines[0])
        self.assertIn("file=30", lines[0])
        self.assertRegex(lines[1], r"^OK ready ")
        self.assertIn("status=downloaded", lines[1])
        self.assertEqual(outcome.path.name, "model.safetensors")

    def test_values_are_single_line_and_secrets_are_redacted(self):
        stream = io.StringIO()
        logger = dwa.StructuredLogger(stream=stream)
        logger.error(
            "failure",
            stage="download",
            message="bad\nurl https://x.test/?token=super-secret",
        )
        lines = stream.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertNotIn("super-secret", lines[0])
        self.assertIn("token=***", lines[0])

    def test_resolved_download_failure_is_exactly_two_lines(self):
        stream = io.StringIO()
        logger = dwa.StructuredLogger(stream=stream)
        resolved = dwa.ResolvedCivitAIResource(
            original="20",
            model_id="10",
            version_id="20",
            file_id="30",
            filename="model.safetensors",
            size_bytes=5,
            sha256="A" * 64,
            file_type="Model",
            file_format="SafeTensor",
        )
        downloader = dwa.CivitAIDownloader("token", tempfile.mkdtemp(), logger=logger)
        downloader.resolve_identifier = lambda _: resolved
        downloader._download_resource = mock.Mock(
            side_effect=dwa.DownloadError("aria2c exited 1")
        )

        ok, path = downloader.download_with_aria2("20", None)

        self.assertFalse(ok)
        self.assertIsNone(path)
        lines = stream.getvalue().splitlines()
        self.assertEqual(len(lines), 2, stream.getvalue())
        self.assertTrue(lines[0].startswith("INFO resolve "))
        self.assertEqual(
            lines[1], 'ERROR failure stage=download message="aria2c exited 1"'
        )

    def test_authentication_failure_is_one_terminal_line(self):
        stream = io.StringIO()
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch("sys.stdout", stream),
        ):
            result = dwa.main(["-m", "20"])

        lines = stream.getvalue().splitlines()
        self.assertEqual(result, 1)
        self.assertEqual(len(lines), 1)
        self.assertRegex(lines[0], r"^ERROR failure stage=auth ")


if __name__ == "__main__":
    unittest.main()
