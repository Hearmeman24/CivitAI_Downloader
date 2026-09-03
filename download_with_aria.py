#!/usr/bin/env python3
"""
CivitAI Model Downloader - Downloads AI models from CivitAI with intelligent file handling.
Supports automatic ZIP extraction, safetensors filtering, and robust error recovery.

Changes:
- Removed duplicate _download_with_url definition.
- Derive filename from Content-Disposition headers when not explicitly provided.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import parse_qs, unquote, urlencode, urlparse

import requests

# Constants
CIVITAI_API_BASE = "https://civitai.com/api"
ARIA2_CONNECTIONS = 8
ARIA2_SPLITS = 8
PROGRESS_INTERVAL = 10
SAFETENSORS_EXT = ".safetensors"
ZIP_EXT = ".zip"
ARIA2_EXT = ".aria2"
MIN_FILE_MB = 1  # basic sanity threshold
CIVITAI_HOSTS = {"civitai.com", "civitai.green", "civitai.red"}

# Status indicators for better UX
STATUS = {
    "success": "✅",
    "error": "❌",
    "warning": "⚠️",
    "info": "🔍",
    "download": "📥",
    "extract": "📦",
    "cleanup": "🗑️",
    "file": "📁",
}

# CivitAI authenticates downloads with a token in the query string, so the
# secret rides along in every URL we build. aria2c echoes that URL in its
# warnings and result summary, and requests embeds it in exception text, so
# anything on its way to the console goes through redact() first.
_TOKEN_RE = re.compile(
    r"((?:[?&](?:token|api_key)=)|(?:Bearer\s+))([^&\s\"'\\]+)", re.IGNORECASE
)


def redact(value) -> str:
    """Mask API tokens in anything about to be printed."""
    return _TOKEN_RE.sub(lambda m: f"{m.group(1)}***", str(value))


class IdentifierError(ValueError):
    """A CivitAI identifier is invalid, ambiguous, or cannot be resolved safely."""


class CivitAIReference:
    """The IDs provided by one supported CivitAI identifier form."""

    def __init__(
        self,
        original: str,
        kind: str,
        model_id: Optional[str] = None,
        version_id: Optional[str] = None,
        file_id: Optional[str] = None,
    ):
        self.original = original
        self.kind = kind
        self.model_id = model_id
        self.version_id = version_id
        self.file_id = file_id


class ResolvedCivitAIResource:
    """A verified CivitAI version and the exact file selected from it."""

    def __init__(
        self,
        original: str,
        model_id: str,
        version_id: str,
        file_id: str,
        filename: str,
    ):
        self.original = original
        self.model_id = model_id
        self.version_id = version_id
        self.file_id = file_id
        self.filename = filename


def _positive_id(value: Optional[str], label: str) -> Optional[str]:
    """Validate and normalize a positive integer ID."""
    if value is None:
        return None
    value = value.strip()
    if not re.fullmatch(r"[1-9]\d*", value):
        raise IdentifierError(f"Invalid CivitAI {label}: expected a positive integer")
    return value


def _first_query_value(query, name: str) -> Optional[str]:
    """Read a query parameter case-insensitively."""
    for key, values in query.items():
        if key.lower() == name.lower() and values:
            return values[0]
    return None


def parse_civitai_reference(value: str) -> CivitAIReference:
    """Parse a model/version ID, AIR, or CivitAI URL without making a request.

    Bare numbers are intentionally classified later because CivitAI's model and
    model-version IDs use separate, overlapping numeric namespaces.
    """
    original = str(value or "").strip()
    if not original:
        raise IdentifierError("A CivitAI ID, AIR, or URL is required")

    # Full AIR as copied by CivitAI, its displayed `civitai:` suffix, and the
    # compact numeric form all carry enough context to select an exact file.
    air_match = re.fullmatch(
        r"(?:urn:air:[^:\s]+:[^:\s]+:)?civitai:"
        r"(?P<model>[1-9]\d*)@(?P<version>[1-9]\d*)"
        r"(?:\+(?P<file>[1-9]\d*))?",
        original,
        flags=re.IGNORECASE,
    )
    if not air_match:
        air_match = re.fullmatch(
            r"(?P<model>[1-9]\d*)@(?P<version>[1-9]\d*)"
            r"(?:\+(?P<file>[1-9]\d*))?",
            original,
        )
    if air_match:
        return CivitAIReference(
            original=original,
            kind="air",
            model_id=air_match.group("model"),
            version_id=air_match.group("version"),
            file_id=air_match.group("file"),
        )

    # Explicit prefixes are the escape hatch when a bare number exists in both
    # of CivitAI's numeric namespaces.
    explicit_match = re.fullmatch(
        r"(?P<kind>model|version)\s*:\s*(?P<id>[1-9]\d*)",
        original,
        flags=re.IGNORECASE,
    )
    if explicit_match:
        kind = explicit_match.group("kind").lower()
        resource_id = explicit_match.group("id")
        return CivitAIReference(
            original=original,
            kind=kind,
            model_id=resource_id if kind == "model" else None,
            version_id=resource_id if kind == "version" else None,
        )

    if re.match(r"^file\s*:", original, flags=re.IGNORECASE):
        raise IdentifierError(
            "CivitAI cannot resolve a file ID alone. Paste the complete AIR "
            "(civitai:model@version+file) or the model page/download URL instead."
        )

    parsed = urlparse(original)
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if parsed.scheme in ("http", "https") and host in CIVITAI_HOSTS:
        query = parse_qs(parsed.query)
        version_id = _positive_id(
            _first_query_value(query, "modelVersionId"), "model version ID"
        )
        file_id = _positive_id(_first_query_value(query, "fileId"), "file ID")

        page_match = re.match(r"^/models/([1-9]\d+)(?:/|$)", parsed.path)
        if page_match:
            model_id = page_match.group(1)
            if file_id and not version_id:
                raise IdentifierError(
                    "A CivitAI URL with a file ID must also include modelVersionId"
                )
            return CivitAIReference(
                original=original,
                kind="version" if version_id else "model",
                model_id=model_id,
                version_id=version_id,
                file_id=file_id,
            )

        version_match = re.match(
            r"^/api/(?:v1/model-versions|download/models)/([1-9]\d+)(?:/|$)",
            parsed.path,
        )
        if version_match:
            return CivitAIReference(
                original=original,
                kind="version",
                version_id=version_match.group(1),
                file_id=file_id,
            )

        raise IdentifierError("Unsupported CivitAI URL; paste a model or download URL")

    if re.fullmatch(r"[1-9]\d*", original):
        return CivitAIReference(original=original, kind="auto", version_id=original)

    raise IdentifierError(
        "Unsupported CivitAI identifier. Paste a model ID, version ID, model URL, "
        "download URL, or complete AIR."
    )


class CivitAIDownloader:
    """Handles downloading and processing of CivitAI model files."""

    def __init__(self, token: str, output_dir: str = "."):
        self.token = token or ""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Header filename helpers ------------------------------------------------

    @staticmethod
    def _parse_content_disposition_filename(header_value: str) -> Optional[str]:
        """
        Parse Content-Disposition to extract filename.
        Supports filename* (RFC 5987) and filename.
        """
        if not header_value:
            return None

        # Try RFC5987: filename*=utf-8''encoded-name
        m = re.search(
            r'filename\*\s*=\s*([^\'";]+)\'\'([^;]+)', header_value, flags=re.IGNORECASE
        )
        if m:
            # charset = m.group(1)  # usually utf-8
            encoded = m.group(2)
            try:
                return unquote(encoded)
            except Exception:
                pass

        # Fallback: filename="..."/filename=...
        m = re.search(r'filename\s*=\s*"([^"]+)"', header_value, flags=re.IGNORECASE)
        if m:
            return m.group(1)

        m = re.search(r"filename\s*=\s*([^;]+)", header_value, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()

        return None

    def _resolve_redirect(self, url: str) -> Tuple[str, Optional[str]]:
        """
        Resolve CivitAI's redirect to get the direct download URL and filename.
        aria2c cannot follow CivitAI's 307 redirects to B2 (B2 returns 403),
        so we resolve the redirect here and pass the final URL to aria2c.
        Returns (resolved_url, filename_or_None).
        """
        try:
            r = requests.get(url, allow_redirects=False, timeout=30)
            if r.status_code in (301, 302, 303, 307, 308):
                resolved = r.headers["Location"]
                # Extract filename from b2ContentDisposition query param
                parsed = urlparse(resolved)
                qs = parse_qs(parsed.query)
                fname = None
                if "b2ContentDisposition" in qs:
                    cd_value = unquote(qs["b2ContentDisposition"][0])
                    fname = self._parse_content_disposition_filename(cd_value)
                if fname:
                    print(f"{STATUS['info']} Server filename: {fname}")
                else:
                    print(
                        f"{STATUS['warning']} Could not extract filename from redirect URL"
                    )
                return resolved, fname
            elif r.status_code == 200:
                # No redirect, extract filename from Content-Disposition
                cd = r.headers.get("Content-Disposition", "")
                fname = self._parse_content_disposition_filename(cd)
                if fname:
                    print(f"{STATUS['info']} Server filename: {fname}")
                return url, fname
            else:
                print(
                    f"{STATUS['warning']} Unexpected status {r.status_code} resolving download URL"
                )
                return url, None
        except requests.RequestException as e:
            print(f"{STATUS['warning']} Could not resolve download URL: {redact(e)}")
            return url, None

    # --- Identifier and metadata resolution -----------------------------------

    def _fetch_metadata(self, path: str, resource_name: str):
        """Fetch one public CivitAI metadata object; return None for a real 404."""
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        url = f"{CIVITAI_API_BASE}/v1/{path}"
        try:
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise IdentifierError(
                    f"CivitAI returned invalid {resource_name} metadata"
                )
            return data
        except requests.RequestException as e:
            raise IdentifierError(
                f"Could not check the CivitAI {resource_name}: {redact(e)}"
            )
        except ValueError as e:
            raise IdentifierError(
                f"CivitAI returned invalid {resource_name} metadata: {redact(e)}"
            )

    def _fetch_version(self, version_id: str):
        data = self._fetch_metadata(f"model-versions/{version_id}", "model version ID")
        if data is None or str(data.get("id")) != str(version_id):
            return None
        return data

    def _fetch_model(self, model_id: str):
        data = self._fetch_metadata(f"models/{model_id}", "model ID")
        if data is None or str(data.get("id")) != str(model_id):
            return None
        return data

    @staticmethod
    def _default_version_id(model_data) -> str:
        versions = model_data.get("modelVersions") or []
        if not versions or not versions[0].get("id"):
            raise IdentifierError(
                f"CivitAI model {model_data.get('id', '')} has no published versions"
            )
        return str(versions[0]["id"])

    @staticmethod
    def _select_version_file(version_data, requested_file_id: Optional[str] = None):
        """Select an exact requested file or the version's primary model file."""
        files = version_data.get("files") or []
        if not files:
            raise IdentifierError(
                f"CivitAI version {version_data.get('id', '')} has no downloadable files"
            )

        if requested_file_id:
            for file_data in files:
                if str(file_data.get("id")) == str(requested_file_id):
                    return file_data
            raise IdentifierError(
                f"CivitAI file {requested_file_id} does not belong to version "
                f"{version_data.get('id', '')}"
            )

        for file_data in files:
            if file_data.get("primary"):
                return file_data
        for file_data in files:
            metadata = file_data.get("metadata") or {}
            if (
                file_data.get("type") == "Model"
                and metadata.get("format") == "SafeTensor"
            ):
                return file_data
        for file_data in files:
            if file_data.get("type") == "Model":
                return file_data
        return files[0]

    def _resolve_version_reference(
        self, reference: CivitAIReference, version_data=None
    ) -> ResolvedCivitAIResource:
        version_id = reference.version_id
        if not version_id:
            raise IdentifierError("A CivitAI model version ID is required")
        if version_data is None:
            version_data = self._fetch_version(version_id)
        if version_data is None:
            raise IdentifierError(f"CivitAI model version {version_id} was not found")

        model_id = str(version_data.get("modelId") or "")
        if not model_id:
            raise IdentifierError(
                f"CivitAI version {version_id} did not identify its parent model"
            )
        if reference.model_id and reference.model_id != model_id:
            raise IdentifierError(
                f"CivitAI version {version_id} belongs to model {model_id}, not "
                f"model {reference.model_id}"
            )

        selected = self._select_version_file(version_data, reference.file_id)
        file_id = str(selected.get("id") or "")
        filename = selected.get("name")
        if not file_id or not filename:
            raise IdentifierError(
                f"CivitAI version {version_id} returned incomplete file metadata"
            )
        return ResolvedCivitAIResource(
            original=reference.original,
            model_id=model_id,
            version_id=version_id,
            file_id=file_id,
            filename=filename,
        )

    def _resolve_model_reference(
        self, reference: CivitAIReference, model_data=None
    ) -> ResolvedCivitAIResource:
        model_id = reference.model_id
        if not model_id:
            raise IdentifierError("A CivitAI model ID is required")
        if model_data is None:
            model_data = self._fetch_model(model_id)
        if model_data is None:
            raise IdentifierError(f"CivitAI model {model_id} was not found")

        version_id = self._default_version_id(model_data)
        print(
            f"{STATUS['info']} Model ID {model_id} resolved to default version {version_id}"
        )
        version_reference = CivitAIReference(
            original=reference.original,
            kind="version",
            model_id=model_id,
            version_id=version_id,
        )
        return self._resolve_version_reference(version_reference)

    def resolve_identifier(self, identifier: str) -> ResolvedCivitAIResource:
        """Resolve a supported input into one verified CivitAI file.

        A bare number is checked as both a model and a version. If it exists in
        both namespaces, stopping is the only safe behavior: choosing either
        could silently download an unrelated file.
        """
        reference = parse_civitai_reference(identifier)
        if reference.kind == "model":
            return self._resolve_model_reference(reference)
        if reference.kind in ("version", "air"):
            return self._resolve_version_reference(reference)

        bare_id = reference.version_id
        version_data = self._fetch_version(bare_id)
        model_data = self._fetch_model(bare_id)
        if version_data is not None and model_data is not None:
            version_parent = (version_data.get("model") or {}).get("name") or "unknown"
            model_name = model_data.get("name") or "unknown"
            raise IdentifierError(
                f"{bare_id} is both a model ID and a version ID. "
                f"Use model:{bare_id} for model '{model_name}', or "
                f"version:{bare_id} for the version belonging to '{version_parent}'."
            )
        if version_data is not None:
            return self._resolve_version_reference(reference, version_data)
        if model_data is not None:
            model_reference = CivitAIReference(
                original=reference.original,
                kind="model",
                model_id=bare_id,
            )
            return self._resolve_model_reference(model_reference, model_data)
        raise IdentifierError(
            f"CivitAI could not find {bare_id} as a model or version ID. "
            "If this is a file ID alone, CivitAI's public API cannot map it back "
            "to its version; paste the complete AIR or model/download URL."
        )

    # --- File utilities ---------------------------------------------------------

    def validate_file(self, file_path: Path) -> Tuple[bool, str]:
        """Validate file existence and check for incomplete downloads."""
        if not file_path.exists():
            return False, "File does not exist"

        if file_path.with_suffix(file_path.suffix + ARIA2_EXT).exists():
            return False, "Incomplete download detected (aria2 control file exists)"

        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb < MIN_FILE_MB:
            return False, f"File suspiciously small ({file_size_mb:.2f}MB)"

        return True, f"File valid ({file_size_mb:.1f}MB)"

    def cleanup_incomplete_download(self, file_path: Path) -> None:
        """Remove incomplete download artifacts."""
        if file_path.exists():
            print(f"{STATUS['cleanup']} Removing incomplete file: {file_path.name}")
            file_path.unlink(missing_ok=True)

        aria2_file = file_path.with_suffix(file_path.suffix + ARIA2_EXT)
        if aria2_file.exists():
            print(f"{STATUS['cleanup']} Removing aria2 control file")
            aria2_file.unlink(missing_ok=True)

    def extract_safetensors_from_zip(
        self, zip_path: Path
    ) -> Tuple[bool, str, Optional[Path]]:
        """
        Extract and keep only safetensors files from ZIP archive.
        Returns (ok, message, last_extracted_path or None)
        """
        print(f"{STATUS['extract']} Extracting: {zip_path.name}")
        temp_dir = self.output_dir / f"temp_extract_{zip_path.stem}"
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                safetensors_in_zip = [
                    n for n in zip_ref.namelist() if n.lower().endswith(SAFETENSORS_EXT)
                ]
                if not safetensors_in_zip:
                    print(
                        f"{STATUS['warning']} No safetensors files found in archive; keeping original ZIP"
                    )
                    return (
                        True,
                        "No safetensors files in archive - keeping original ZIP",
                        None,
                    )

                temp_dir.mkdir(exist_ok=True)
                for file_name in safetensors_in_zip:
                    zip_ref.extract(file_name, temp_dir)

                moved_count = 0
                last_moved: Optional[Path] = None
                for extracted_file in temp_dir.rglob(f"*{SAFETENSORS_EXT}"):
                    dest_file = self._get_unique_filename(
                        self.output_dir / extracted_file.name
                    )
                    shutil.move(str(extracted_file), str(dest_file))
                    print(f"{STATUS['file']} Extracted: {dest_file.name}")
                    moved_count += 1
                    last_moved = dest_file

                print(f"{STATUS['cleanup']} Removing temporary files and original ZIP")
                shutil.rmtree(temp_dir, ignore_errors=True)
                zip_path.unlink(missing_ok=True)
                return True, f"Extracted {moved_count} safetensors file(s)", last_moved

        except zipfile.BadZipFile:
            return False, "Corrupted or invalid ZIP file", None
        except Exception as e:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            return False, f"Extraction error: {e}", None

    def _get_unique_filename(self, file_path: Path) -> Path:
        """Generate unique filename if conflict exists."""
        if not file_path.exists():
            return file_path
        counter = 1
        while True:
            new_path = (
                file_path.parent / f"{file_path.stem}_{counter}{file_path.suffix}"
            )
            if not new_path.exists():
                return new_path
            counter += 1

    def process_downloaded_file(
        self, file_path: Path
    ) -> Tuple[bool, str, Optional[Path]]:
        """Process downloaded file based on its type. Returns (ok, msg, final_path)."""
        print(f"{STATUS['info']} Processing file: {file_path.name}")
        if not file_path.exists():
            return False, "Downloaded file not found", None

        suffix = file_path.suffix.lower()
        if suffix == ZIP_EXT:
            ok, msg, last_path = self.extract_safetensors_from_zip(file_path)
            return ok, msg, last_path
        elif suffix == SAFETENSORS_EXT:
            print(f"{STATUS['success']} File is already in safetensors format")
            return True, "File ready to use", file_path
        else:
            print(f"{STATUS['info']} File type: {file_path.suffix}")
            return True, "File downloaded successfully", file_path

    # --- Download core ----------------------------------------------------------

    def _run_aria2c(self, cmd: list) -> int:
        """Run aria2c, streaming its output with any token masked.

        aria2c prints the request URI in its warnings and in the
        --download-result summary, so its output cannot go straight to the
        terminal while the URL carries the token. Streaming line by line keeps
        the periodic progress summary live.
        """
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            print(redact(line), end="")
        return proc.wait()

    def _download_with_url(
        self, download_url: str, prefer_filename: Optional[str], force: bool = False
    ) -> Tuple[bool, Optional[Path]]:
        """
        Download file using aria2c with the given URL.
        Resolves CivitAI redirects first since aria2c gets 403 following them.
        Returns (ok, downloaded_path or None).
        """
        # Resolve redirect to get direct B2 URL and filename
        resolved_url, redirect_filename = self._resolve_redirect(download_url)
        filename = redirect_filename or prefer_filename or "download.bin"

        file_path = self.output_dir / filename

        aria2_control = file_path.with_suffix(file_path.suffix + ARIA2_EXT)

        # --force means "ignore whatever is on disk and start fresh": wipe the
        # partial AND the .aria2 control file so --continue has nothing stale to
        # resume. Without this, force silently resumes and can't recover a wedged
        # download (e.g. a control file whose length no longer matches the server).
        if force:
            self.cleanup_incomplete_download(file_path)

        # Check if file already exists and is valid (unless force is True)
        resuming = False
        if not force and file_path.exists():
            is_valid, message = self.validate_file(file_path)
            if is_valid:
                print(
                    f"{STATUS['success']} File already exists and is valid: {file_path.name} ({message})"
                )
                return True, file_path
            elif aria2_control.exists():
                # A partial file WITH its .aria2 control file is resumable:
                # aria2 verified the completed byte ranges, so --continue finishes
                # it safely. Deleting here would defeat the whole point of resume.
                print(
                    f"{STATUS['download']} Resuming interrupted download: {file_path.name}"
                )
                resuming = True
            else:
                # Orphaned partial with no resume state -> can't trust it, start clean.
                print(
                    f"{STATUS['warning']} Existing file is invalid: {message}. Re-downloading..."
                )
                self.cleanup_incomplete_download(file_path)
        elif not force and aria2_control.exists():
            # Control file with no data file: aria2c aborts on --continue against
            # this, and it never self-heals. Drop the stale control and start fresh.
            print(
                f"{STATUS['cleanup']} Removing stale aria2 control file (no data file)"
            )
            aria2_control.unlink(missing_ok=True)

        # Only use unique filename generation if we're actually going to download,
        # not forcing, and not resuming (resuming must reuse the exact same name).
        if not force and not resuming:
            file_path = self._get_unique_filename(
                file_path
            )  # avoid accidental overwrite collisions

        print(f"{STATUS['info']} Expected filename: {file_path.name}")

        # Build aria2 command with the resolved (direct) URL
        cmd = [
            "aria2c",
            f"--max-connection-per-server={ARIA2_CONNECTIONS}",
            f"--split={ARIA2_SPLITS}",
            "--continue=true",
            "--file-allocation=none",
            "--auto-file-renaming=false",
            "--allow-overwrite=true",
            f"--summary-interval={PROGRESS_INTERVAL}",
            "--console-log-level=warn",
            "--download-result=full",
            f"--dir={self.output_dir}",
            f"--out={file_path.name}",
            resolved_url,
        ]

        print(f"{STATUS['download']} Downloading {file_path.name}")
        print(f"{STATUS['info']} Using {ARIA2_CONNECTIONS} connections")

        try:
            returncode = self._run_aria2c(cmd)
            if returncode != 0:
                print(f"{STATUS['error']} Download failed: aria2c exited {returncode}")
                return False, None

            # Validate expected file or discover last modified in case server changed it
            print(f"{STATUS['info']} Checking for downloaded files...")
            all_files = list(self.output_dir.glob("*"))
            print(f"{STATUS['info']} Files in directory: {[f.name for f in all_files]}")

            actual = file_path if file_path.exists() else None
            if not actual:
                print(f"{STATUS['warning']} Expected file not found: {file_path.name}")
                recent_files = [
                    f
                    for f in all_files
                    if f.is_file() and (time.time() - f.stat().st_mtime) < 120
                ]
                if recent_files:
                    actual = max(recent_files, key=lambda f: f.stat().st_mtime)
                    print(
                        f"{STATUS['info']} Using most recent file as downloaded: {actual.name}"
                    )

            if not actual:
                print(f"{STATUS['error']} Could not locate a downloaded file")
                return False, None

            is_valid, message = self.validate_file(actual)
            if not is_valid:
                print(f"{STATUS['error']} Download validation failed: {message}")
                return False, None

            print(f"{STATUS['success']} Download complete: {message}")
            return True, actual

        except FileNotFoundError:
            print(f"{STATUS['error']} aria2c not found. Please install aria2.")
            print("  Ubuntu/Debian: sudo apt-get install aria2")
            print("  macOS: brew install aria2")
            print("  Windows: Download from https://aria2.github.io/")
            return False, None

    def download_with_aria2(
        self, identifier: str, prefer_filename: Optional[str], force: bool = False
    ) -> Tuple[bool, Optional[Path]]:
        """
        Resolve a CivitAI identifier and download its exact selected file.
        prefer_filename: user-supplied target name (may be None).
        Returns (ok, final_path or None).
        """
        resource = self.resolve_identifier(identifier)
        print(
            f"{STATUS['info']} Resolved model {resource.model_id}, "
            f"version {resource.version_id}, file {resource.file_id}"
        )

        # CivitAI's fileId selector is the only deterministic selection when a
        # version has multiple Model files with the same metadata or filename.
        params = {"fileId": resource.file_id}
        if self.token:
            params["token"] = self.token
        download_url = (
            f"{CIVITAI_API_BASE}/download/models/{resource.version_id}"
            f"?{urlencode(params)}"
        )
        target_name = prefer_filename or resource.filename
        ok, path = self._download_with_url(download_url, target_name, force)
        if ok and path:
            ok2, msg, final_path = self.process_downloaded_file(path)
            if ok2:
                print(f"{STATUS['success']} {msg}")
                return True, final_path or path
            print(f"{STATUS['error']} Processing failed: {msg}")

        print(f"{STATUS['error']} Download failed")
        return False, None


def get_token(args_token: Optional[str]) -> str:
    """Retrieve CivitAI token from environment or arguments."""
    token = os.getenv("CIVITAI_TOKEN") or os.getenv("civitai_token")
    if token:
        print(f"{STATUS['success']} Using token from environment variable")
        return token
    elif args_token:
        print(f"{STATUS['success']} Using token from command line")
        return args_token
    else:
        print(f"{STATUS['error']} No CivitAI token provided")
        print("  Set CIVITAI_TOKEN environment variable or use --token argument")
        sys.exit(1)


def main():
    """Main entry point for the downloader."""
    parser = argparse.ArgumentParser(
        description="Download AI models from CivitAI with intelligent file handling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -m 3268303                   # Auto-detect a version or model ID
  %(prog)s -m model:2834417             # Explicit model ID; use its default version
  %(prog)s -m version:3268303           # Explicit model-version ID
  %(prog)s -m 'civitai:2834417@3268303+3152083'  # Exact AIR file
  %(prog)s -m 'https://civitai.com/models/2834417?modelVersionId=3268303'
        """,
    )

    parser.add_argument(
        "-m",
        "--identifier",
        "--model-id",
        dest="identifier",
        required=True,
        metavar="IDENTIFIER",
        help="CivitAI model/version ID, model URL, download URL, or AIR",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=".",
        help="Output directory (default: current directory)",
    )
    parser.add_argument(
        "--token", help="CivitAI API token (or set CIVITAI_TOKEN env variable)"
    )
    parser.add_argument(
        "--filename", help="Override filename (default: taken from server headers)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if valid file exists",
    )

    args = parser.parse_args()

    try:
        token = get_token(args.token)
        downloader = CivitAIDownloader(token, args.output)

        # Prefer user-supplied name; otherwise we’ll derive from headers at download time.
        prefer_filename = args.filename
        if prefer_filename:
            print(f"{STATUS['info']} Using custom filename: {prefer_filename}")

        ok, final_path = downloader.download_with_aria2(
            args.identifier, prefer_filename, force=args.force
        )
        if ok:
            if final_path and final_path.exists():
                print(f"{STATUS['success']} Model ready at: {final_path}")
            else:
                # fallback: pick most recent .safetensors in output
                safes = sorted(
                    downloader.output_dir.glob("*.safetensors"),
                    key=lambda p: p.stat().st_mtime,
                )
                if safes:
                    print(f"{STATUS['success']} Model ready at: {safes[-1]}")
                else:
                    print(f"{STATUS['success']} Download completed successfully")
        else:
            sys.exit(1)

    except IdentifierError as e:
        print(f"{STATUS['error']} {redact(e)}")
        sys.exit(2)
    except KeyboardInterrupt:
        print(f"\n{STATUS['warning']} Download interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"{STATUS['error']} Unexpected error: {redact(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
