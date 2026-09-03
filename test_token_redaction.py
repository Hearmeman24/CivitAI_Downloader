#!/usr/bin/env python3
"""Credential secrecy tests for logs and child-process boundaries."""

import io
import os
import tempfile
import unittest
from unittest import mock

import download_with_aria as dwa

SECRET = "sk-civitai-DEADBEEF0123456789"


class TokenRedactionTests(unittest.TestCase):
    def test_redact_masks_supported_query_and_header_credentials(self):
        samples = (
            f"https://x.test/?token={SECRET}",
            f"https://x.test/?api_key={SECRET}",
            f"https://x.test/?Authorization={SECRET}",
            f"https://x.test/?X-Amz-Signature={SECRET}",
            f"Authorization: Bearer {SECRET}",
        )
        for sample in samples:
            with self.subTest(sample=sample):
                self.assertNotIn(SECRET, dwa.redact(sample))

    def test_logger_keeps_errors_single_line_without_secret(self):
        stream = io.StringIO()
        logger = dwa.StructuredLogger(stream)
        logger.error(
            "failure",
            stage="download",
            message=f"failed\nhttps://x.test/?token={SECRET}",
        )
        self.assertEqual(len(stream.getvalue().splitlines()), 1)
        self.assertNotIn(SECRET, stream.getvalue())

    def test_aria_receives_url_over_stdin_and_not_civitai_environment(self):
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
                captured["env"] = kwargs["env"]
                self.stdin = Sink()
                captured["stdin"] = self.stdin
                self.stdout = io.StringIO(
                    f"failed https://x.test/?Authorization={SECRET}\n"
                )

            def wait(self):
                return 1

        source = f"https://x.test/?Authorization={SECRET}"
        with (
            mock.patch.dict(os.environ, {"CIVITAI_TOKEN": SECRET}),
            mock.patch.object(dwa.subprocess, "Popen", FakeProcess),
        ):
            result = dwa.CivitAIDownloader("token", tempfile.mkdtemp())._run_aria2c(
                ["aria2c", "--input-file=-"], source
            )

        self.assertNotIn(source, captured["cmd"])
        self.assertNotIn("CIVITAI_TOKEN", captured["env"])
        self.assertEqual(captured["stdin"].value, source + "\n")
        self.assertNotIn(SECRET, " ".join(result.diagnostic_tail))


if __name__ == "__main__":
    unittest.main()
