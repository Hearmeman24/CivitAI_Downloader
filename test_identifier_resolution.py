#!/usr/bin/env python3
"""Behavior checks for CivitAI model/version/file identifier resolution."""

import hashlib
import io
import tempfile
import unittest
from urllib.parse import parse_qs, urlparse

import download_with_aria as dwa

MODEL_ID = "2834417"
VERSION_ID = "3268303"
FILE_ID = "3152083"
FILENAME = "HMNSFW-AIO-V2.5.safetensors"
FILE_BYTES = b"model payload"
FILE_SHA256 = hashlib.sha256(FILE_BYTES).hexdigest().upper()


def version_data(model_id=MODEL_ID, version_id=VERSION_ID, file_id=FILE_ID):
    return {
        "id": int(version_id),
        "modelId": int(model_id),
        "name": "V2.5",
        "model": {"name": "HMNSFW - AIO Sex LoRA", "type": "LORA"},
        "files": [
            {
                "id": int(file_id),
                "name": FILENAME,
                "type": "Model",
                "primary": True,
                "metadata": {"format": "SafeTensor", "fp": "fp16"},
                "sizeKB": len(FILE_BYTES) / 1024,
                "hashes": {"SHA256": FILE_SHA256},
            }
        ],
    }


def model_data(model_id=MODEL_ID, version_id=VERSION_ID):
    return {
        "id": int(model_id),
        "name": "HMNSFW - AIO Sex LoRA",
        "modelVersions": [{"id": int(version_id), "name": "V2.5"}],
    }


class ParseIdentifierTests(unittest.TestCase):
    def assert_reference(self, raw, model=None, version=None, file_id=None, kind=None):
        ref = dwa.parse_civitai_reference(raw)
        self.assertEqual(ref.model_id, model)
        self.assertEqual(ref.version_id, version)
        self.assertEqual(ref.file_id, file_id)
        if kind is not None:
            self.assertEqual(ref.kind, kind)

    def test_full_air_and_short_air_keep_exact_file_identity(self):
        self.assert_reference(
            "urn:air:minimaxh3:lora:civitai:2834417@3268303+3152083",
            MODEL_ID,
            VERSION_ID,
            FILE_ID,
            "air",
        )
        self.assert_reference(
            "civitai:2834417@3268303+3152083",
            MODEL_ID,
            VERSION_ID,
            FILE_ID,
            "air",
        )
        self.assert_reference(
            "2834417@3268303+3152083",
            MODEL_ID,
            VERSION_ID,
            FILE_ID,
            "air",
        )

    def test_civitai_page_and_download_urls_are_accepted(self):
        self.assert_reference(
            "https://civitai.com/models/2834417/hmnsfw?modelVersionId=3268303",
            MODEL_ID,
            VERSION_ID,
            None,
            "version",
        )
        self.assert_reference(
            "https://civitai.com/api/download/models/3268303?fileId=3152083",
            None,
            VERSION_ID,
            FILE_ID,
            "version",
        )

    def test_explicit_prefixes_and_bare_number_are_accepted(self):
        self.assert_reference("model:2834417", MODEL_ID, None, None, "model")
        self.assert_reference("version:3268303", None, VERSION_ID, None, "version")
        self.assert_reference("3268303", None, VERSION_ID, None, "auto")

    def test_bad_or_context_free_file_input_is_rejected_locally(self):
        for raw in ("", "not-an-id", "file:3152083", "2834417@3268303+nope"):
            with self.subTest(raw=raw), self.assertRaises(dwa.IdentifierError):
                dwa.parse_civitai_reference(raw)


class ResolveIdentifierTests(unittest.TestCase):
    def downloader(self):
        return dwa.CivitAIDownloader(
            token="test-token",
            output_dir=tempfile.mkdtemp(),
            logger=dwa.StructuredLogger(io.StringIO()),
        )

    def test_model_id_resolves_to_default_version_and_primary_file(self):
        dl = self.downloader()
        dl._fetch_model = lambda value: model_data() if value == MODEL_ID else None
        dl._fetch_version = lambda value: (
            version_data() if value == VERSION_ID else None
        )

        resolved = dl.resolve_identifier("model:2834417")

        self.assertEqual(resolved.model_id, MODEL_ID)
        self.assertEqual(resolved.version_id, VERSION_ID)
        self.assertEqual(resolved.file_id, FILE_ID)
        self.assertEqual(resolved.filename, FILENAME)

    def test_version_id_resolves_to_its_primary_file(self):
        dl = self.downloader()
        dl._fetch_model = lambda value: None
        dl._fetch_version = lambda value: (
            version_data() if value == VERSION_ID else None
        )

        resolved = dl.resolve_identifier("version:3268303")

        self.assertEqual(resolved.model_id, MODEL_ID)
        self.assertEqual(resolved.version_id, VERSION_ID)
        self.assertEqual(resolved.file_id, FILE_ID)

    def test_full_air_selects_and_validates_the_exact_file(self):
        dl = self.downloader()
        dl._fetch_model = lambda value: None
        dl._fetch_version = lambda value: (
            version_data() if value == VERSION_ID else None
        )

        resolved = dl.resolve_identifier(
            "urn:air:minimaxh3:lora:civitai:2834417@3268303+3152083"
        )

        self.assertEqual(resolved.file_id, FILE_ID)
        self.assertEqual(resolved.filename, FILENAME)

    def test_air_rejects_a_model_or_file_that_does_not_belong_to_version(self):
        dl = self.downloader()
        dl._fetch_model = lambda value: None
        dl._fetch_version = lambda value: version_data()

        with self.assertRaisesRegex(dwa.IdentifierError, "belongs to model"):
            dl.resolve_identifier("civitai:999@3268303+3152083")
        with self.assertRaisesRegex(dwa.IdentifierError, "file 999"):
            dl.resolve_identifier("civitai:2834417@3268303+999")

    def test_missing_published_sha256_fails_closed(self):
        dl = self.downloader()
        unverified = version_data()
        unverified["files"][0]["hashes"] = {}
        dl._fetch_version = lambda value: unverified

        with self.assertRaisesRegex(dwa.IntegrityError, "no valid SHA-256"):
            dl.resolve_identifier("version:3268303")

    def test_unambiguous_bare_model_or_version_ids_just_work(self):
        dl = self.downloader()
        dl._fetch_model = lambda value: model_data() if value == MODEL_ID else None
        dl._fetch_version = lambda value: (
            version_data() if value in (MODEL_ID, VERSION_ID) else None
        )

        # VERSION_ID exists only as a version.
        resolved_version = dl.resolve_identifier(VERSION_ID)
        self.assertEqual(resolved_version.version_id, VERSION_ID)

        # MODEL_ID exists as both in this fixture, so it must not guess.
        with self.assertRaisesRegex(
            dwa.IdentifierError, "both a model ID and a version ID"
        ):
            dl.resolve_identifier(MODEL_ID)

        # When the same model ID is not also a version, it resolves its default.
        dl._fetch_version = lambda value: (
            version_data() if value == VERSION_ID else None
        )
        resolved_model = dl.resolve_identifier(MODEL_ID)
        self.assertEqual(resolved_model.version_id, VERSION_ID)

    def test_unknown_bare_number_explains_the_file_id_limitation(self):
        dl = self.downloader()
        dl._fetch_model = lambda value: None
        dl._fetch_version = lambda value: None

        with self.assertRaisesRegex(dwa.IdentifierError, "file ID alone"):
            dl.resolve_identifier(FILE_ID)

    def test_exact_file_download_uses_civitai_file_id_selector(self):
        dl = self.downloader()
        dl.resolve_identifier = lambda value: dwa.ResolvedCivitAIResource(
            original=value,
            model_id=MODEL_ID,
            version_id=VERSION_ID,
            file_id=FILE_ID,
            filename=FILENAME,
            size_bytes=len(FILE_BYTES),
            sha256=FILE_SHA256,
            file_type="Model",
            file_format="SafeTensor",
        )
        calls = []

        def fake_resolve(url):
            calls.append(url)
            return "https://files.example/signed"

        def fake_aria(cmd, source):
            out_name = next(
                item.split("=", 1)[1] for item in cmd if item.startswith("--out=")
            )
            (dl.output_dir / out_name).write_bytes(FILE_BYTES)
            return dwa.AriaResult(0, ())

        dl._resolve_download_url = fake_resolve
        dl._run_aria2c = fake_aria

        ok, path = dl.download_with_aria2("civitai:2834417@3268303+3152083", None)

        self.assertTrue(ok)
        self.assertEqual(path.name, FILENAME)
        self.assertEqual(len(calls), 1)
        parsed = urlparse(calls[0])
        self.assertEqual(parsed.path, "/api/download/models/3268303")
        self.assertEqual(parse_qs(parsed.query)["fileId"], [FILE_ID])
        self.assertEqual(parse_qs(parsed.query)["token"], ["test-token"])


if __name__ == "__main__":
    unittest.main()
