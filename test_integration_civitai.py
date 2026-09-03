#!/usr/bin/env python3
"""Opt-in end-to-end test against CivitAI and aria2c.

Run with:
    CIVITAI_INTEGRATION=1 CIVITAI_TOKEN=... python3 -m unittest -v test_integration_civitai
"""

import io
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import download_with_aria as dwa

AIR = "urn:air:minimaxh3:lora:civitai:2834417@3268303+3152083"
EXPECTED_SIZE = 86040232
EXPECTED_SHA256 = "A07732A84FD733085EB5D910F602F918FA7A3658117116927E4329F5951A9D2D"


@unittest.skipUnless(
    os.getenv("CIVITAI_INTEGRATION") == "1", "live integration disabled"
)
class LiveCivitAIIntegrationTests(unittest.TestCase):
    def test_exact_air_download_is_verified_and_logs_two_lines(self):
        token = os.getenv("CIVITAI_TOKEN") or os.getenv("civitai_token")
        if not token:
            self.skipTest("CIVITAI_TOKEN is not set")
        if not shutil.which("aria2c"):
            self.skipTest("aria2c is not installed")

        stream = io.StringIO()
        with tempfile.TemporaryDirectory() as td:
            with redirect_stdout(stream):
                returncode = dwa.main(["-m", AIR, "-o", td])
            self.assertEqual(returncode, 0)

            path = Path(td) / "HMNSFW-AIO-V2.5.safetensors"
            self.assertTrue(path.exists())
            self.assertEqual(path.stat().st_size, EXPECTED_SIZE)
            self.assertEqual(dwa.CivitAIDownloader._sha256(path), EXPECTED_SHA256)

            first_lines = stream.getvalue().splitlines()
            self.assertEqual(len(first_lines), 2, stream.getvalue())
            self.assertTrue(first_lines[0].startswith("INFO resolve "))
            self.assertTrue(first_lines[1].startswith("OK ready status=downloaded "))

            stream.seek(0)
            stream.truncate(0)
            with redirect_stdout(stream):
                returncode = dwa.main(["-m", AIR, "-o", td])
            self.assertEqual(returncode, 0)

        lines = stream.getvalue().splitlines()
        self.assertEqual(len(lines), 2, stream.getvalue())
        self.assertTrue(lines[0].startswith("INFO resolve "))
        self.assertTrue(lines[1].startswith("OK ready status=cached "))


if __name__ == "__main__":
    unittest.main()
