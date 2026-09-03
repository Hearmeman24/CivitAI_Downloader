#!/usr/bin/env python3
"""Secure, resumable CivitAI model downloader with concise structured logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil

# aria2c is the intentional fixed executable; no shell is involved.
import subprocess  # nosec B404
import sys
import tempfile
import unicodedata
import zipfile
from collections import deque
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import IO
from urllib.parse import parse_qs, urlencode, urljoin, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

CIVITAI_API_BASE = "https://civitai.com/api"
CIVITAI_HOSTS = {"civitai.com", "civitai.green", "civitai.red"}
ARIA2_CONNECTIONS = 8
ARIA2_SPLITS = 8
ARIA2_EXT = ".aria2"
SAFETENSORS_EXT = ".safetensors"
ZIP_EXT = ".zip"
HTTP_TIMEOUT = (10, 30)
HTTP_RETRIES = 4
MAX_RETRY_AFTER_SECONDS = 30
DISK_RESERVE_BYTES = 64 * 1024 * 1024
HASH_CHUNK_BYTES = 4 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 512
MAX_ARCHIVE_EXPANDED_BYTES = 64 * 1024 * 1024 * 1024
MAX_ARCHIVE_COMPRESSION_RATIO = 200
MAX_FILENAME_BYTES = 240

_TOKEN_RE = re.compile(
    r"((?:[?&](?:token|api_key|authorization|x-amz-signature|"
    r"x-amz-credential|x-amz-security-token)=)|(?:Bearer\s+))([^&\s\"'\\]+)",
    re.IGNORECASE,
)
_SAFE_LOG_VALUE_RE = re.compile(r"^[A-Za-z0-9._:/@+,=-]+$")
_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


def redact(value: object) -> str:
    """Mask CivitAI-style API tokens before text reaches a user or log."""
    return _TOKEN_RE.sub(lambda match: f"{match.group(1)}***", str(value))


class StructuredLogger:
    """One-line key/value events designed for both people and simple parsers."""

    def __init__(self, stream: IO[str] | None = None):
        self.stream = stream or sys.stdout

    @staticmethod
    def _value(value: object) -> str:
        cleaned = redact(value).replace("\r", r"\r").replace("\n", r"\n")
        if cleaned and _SAFE_LOG_VALUE_RE.fullmatch(cleaned):
            return cleaned
        return json.dumps(cleaned, ensure_ascii=False)

    def emit(self, level: str, event: str, **fields: object) -> None:
        parts = [level, event]
        parts.extend(f"{key}={self._value(value)}" for key, value in fields.items())
        print(" ".join(parts), file=self.stream, flush=True)

    def info(self, event: str, **fields: object) -> None:
        self.emit("INFO", event, **fields)

    def success(self, event: str, **fields: object) -> None:
        self.emit("OK", event, **fields)

    def error(self, event: str, **fields: object) -> None:
        self.emit("ERROR", event, **fields)


class DownloaderError(RuntimeError):
    """Base class for concise user-facing failures."""

    stage = "download"

    def __init__(self, message: str, *, stage: str | None = None):
        super().__init__(message)
        if stage is not None:
            self.stage = stage


class IdentifierError(DownloaderError, ValueError):
    stage = "resolve"


class AuthenticationError(DownloaderError):
    stage = "auth"


class MetadataError(DownloaderError):
    stage = "metadata"


class DownloadError(DownloaderError):
    stage = "download"


class IntegrityError(DownloaderError):
    stage = "verify"


class UnsafePathError(DownloaderError):
    stage = "path"


class ArchiveSafetyError(DownloaderError):
    stage = "extract"


class ResourceBusyError(DownloaderError):
    stage = "lock"


class DependencyError(DownloaderError):
    stage = "dependency"


@dataclass(frozen=True)
class CivitAIReference:
    original: str
    kind: str
    model_id: str | None = None
    version_id: str | None = None
    file_id: str | None = None


@dataclass(frozen=True)
class ResolvedCivitAIResource:
    original: str
    model_id: str
    version_id: str
    file_id: str
    filename: str
    size_bytes: int
    sha256: str
    file_type: str
    file_format: str


@dataclass(frozen=True)
class FileVerification:
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class AriaResult:
    returncode: int
    diagnostic_tail: tuple[str, ...]


@dataclass(frozen=True)
class DownloadOutcome:
    path: Path
    artifacts: tuple[Path, ...]
    status: str
    size_bytes: int
    sha256: str


def _positive_id(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if not re.fullmatch(r"[1-9]\d*", value):
        raise IdentifierError(f"invalid CivitAI {label}; expected a positive integer")
    return value


def _first_query_value(query: dict, name: str) -> str | None:
    for key, values in query.items():
        if key.lower() == name.lower() and values:
            return values[0]
    return None


def parse_civitai_reference(value: str) -> CivitAIReference:
    """Parse a model/version ID, AIR, or CivitAI URL without making a request."""
    original = str(value or "").strip()
    if not original:
        raise IdentifierError("a CivitAI ID, AIR, or URL is required")

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
            "a file ID alone cannot be resolved; paste the complete AIR or CivitAI URL"
        )

    parsed = urlparse(original)
    host = (parsed.hostname or "").lower()
    host = host.removeprefix("www.")
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
                    "a CivitAI URL with fileId must also include modelVersionId"
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
        raise IdentifierError("unsupported CivitAI URL; paste a model or download URL")

    if re.fullmatch(r"[1-9]\d*", original):
        return CivitAIReference(original=original, kind="auto", version_id=original)
    raise IdentifierError(
        "unsupported identifier; paste a model ID, version ID, CivitAI URL, or AIR"
    )


class BoundedRetry(Retry):
    """Honor Retry-After without allowing an unbounded server-directed sleep."""

    def get_retry_after(self, response):
        retry_after = super().get_retry_after(response)
        if retry_after is None:
            return None
        return min(retry_after, MAX_RETRY_AFTER_SECONDS)

    def get_backoff_time(self):
        return min(super().get_backoff_time(), MAX_RETRY_AFTER_SECONDS)


def build_http_session() -> requests.Session:
    """Build one pooled session with bounded idempotent retries."""
    retry = BoundedRetry(
        total=HTTP_RETRIES,
        connect=HTTP_RETRIES,
        read=HTTP_RETRIES,
        status=HTTP_RETRIES,
        backoff_factor=0.5,
        status_forcelist=frozenset({429, 500, 502, 503, 504}),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "Hearmeman24-CivitAI-Downloader/2"})
    return session


class CivitAIDownloader:
    """Resolve, transfer, verify, and safely process one CivitAI resource."""

    def __init__(
        self,
        token: str,
        output_dir: os.PathLike[str] | str = ".",
        *,
        logger: StructuredLogger | None = None,
        session: requests.Session | None = None,
    ):
        self.token = token or ""
        requested_output = Path(output_dir).expanduser()
        if any(ord(character) < 32 for character in str(requested_output)):
            raise UnsafePathError("output directory contains control characters")
        requested_output.mkdir(parents=True, exist_ok=True)
        self.output_dir = requested_output.resolve()
        if not self.output_dir.is_dir():
            raise UnsafePathError(f"output is not a directory: {self.output_dir}")
        self.logger = logger or StructuredLogger()
        self.session = session or build_http_session()

    def _fetch_metadata(self, path: str, resource_name: str) -> dict | None:
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        url = f"{CIVITAI_API_BASE}/v1/{path}"
        response: requests.Response | None = None
        try:
            response = self.session.get(url, headers=headers, timeout=HTTP_TIMEOUT)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise MetadataError(
                    f"CivitAI returned invalid {resource_name} metadata"
                )
            return data
        except requests.RequestException as exc:
            raise MetadataError(
                f"could not fetch CivitAI {resource_name}: {redact(exc)}"
            ) from exc
        except ValueError as exc:
            raise MetadataError(
                f"CivitAI returned invalid {resource_name} JSON"
            ) from exc
        finally:
            if response is not None:
                response.close()

    def _fetch_version(self, version_id: str) -> dict | None:
        data = self._fetch_metadata(f"model-versions/{version_id}", "model version")
        if data is None or str(data.get("id")) != str(version_id):
            return None
        return data

    def _fetch_model(self, model_id: str) -> dict | None:
        data = self._fetch_metadata(f"models/{model_id}", "model")
        if data is None or str(data.get("id")) != str(model_id):
            return None
        return data

    @staticmethod
    def _default_version_id(model_data: dict) -> str:
        versions = model_data.get("modelVersions") or []
        if not versions or not versions[0].get("id"):
            raise IdentifierError(
                f"CivitAI model {model_data.get('id', '')} has no published versions"
            )
        return str(versions[0]["id"])

    @staticmethod
    def _select_version_file(
        version_data: dict, requested_file_id: str | None = None
    ) -> dict:
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

    @staticmethod
    def _resource_from_file(
        reference: CivitAIReference, version_data: dict, selected: dict
    ) -> ResolvedCivitAIResource:
        version_id = str(version_data.get("id") or "")
        model_id = str(version_data.get("modelId") or "")
        file_id = str(selected.get("id") or "")
        filename = str(selected.get("name") or "")
        hashes = selected.get("hashes") or {}
        sha256 = str(hashes.get("SHA256") or "").upper()
        size_kb = selected.get("sizeKB")
        metadata = selected.get("metadata") or {}
        if not model_id:
            raise IdentifierError(f"CivitAI version {version_id} has no parent model")
        if reference.model_id and reference.model_id != model_id:
            raise IdentifierError(
                f"CivitAI version {version_id} belongs to model {model_id}, not "
                f"model {reference.model_id}"
            )
        if not file_id or not filename:
            raise IdentifierError(
                f"CivitAI version {version_id} returned incomplete file metadata"
            )
        if not re.fullmatch(r"[A-F0-9]{64}", sha256):
            raise IntegrityError(
                f"CivitAI file {file_id} has no valid SHA-256; refusing unverified download"
            )
        try:
            size_bytes = round(float(size_kb) * 1024)
        except (TypeError, ValueError) as exc:
            raise IntegrityError(
                f"CivitAI file {file_id} has no valid expected size"
            ) from exc
        if size_bytes <= 0:
            raise IntegrityError(f"CivitAI file {file_id} has an invalid expected size")
        return ResolvedCivitAIResource(
            original=reference.original,
            model_id=model_id,
            version_id=version_id,
            file_id=file_id,
            filename=filename,
            size_bytes=size_bytes,
            sha256=sha256,
            file_type=str(selected.get("type") or "unknown"),
            file_format=str(metadata.get("format") or "unknown"),
        )

    def _resolve_version_reference(
        self, reference: CivitAIReference, version_data: dict | None = None
    ) -> ResolvedCivitAIResource:
        version_id = reference.version_id
        if not version_id:
            raise IdentifierError("a CivitAI model version ID is required")
        version_data = version_data or self._fetch_version(version_id)
        if version_data is None:
            raise IdentifierError(f"CivitAI model version {version_id} was not found")
        selected = self._select_version_file(version_data, reference.file_id)
        return self._resource_from_file(reference, version_data, selected)

    def _resolve_model_reference(
        self, reference: CivitAIReference, model_data: dict | None = None
    ) -> ResolvedCivitAIResource:
        model_id = reference.model_id
        if not model_id:
            raise IdentifierError("a CivitAI model ID is required")
        model_data = model_data or self._fetch_model(model_id)
        if model_data is None:
            raise IdentifierError(f"CivitAI model {model_id} was not found")
        version_reference = CivitAIReference(
            original=reference.original,
            kind="version",
            model_id=model_id,
            version_id=self._default_version_id(model_data),
        )
        return self._resolve_version_reference(version_reference)

    def resolve_identifier(self, identifier: str) -> ResolvedCivitAIResource:
        reference = parse_civitai_reference(identifier)
        if reference.kind == "model":
            return self._resolve_model_reference(reference)
        if reference.kind in ("version", "air"):
            return self._resolve_version_reference(reference)

        bare_id = reference.version_id
        if bare_id is None:
            raise IdentifierError("a CivitAI ID is required")
        version_data = self._fetch_version(bare_id)
        model_data = self._fetch_model(bare_id)
        if version_data is not None and model_data is not None:
            version_parent = (version_data.get("model") or {}).get("name") or "unknown"
            model_name = model_data.get("name") or "unknown"
            raise IdentifierError(
                f"{bare_id} is both a model ID and a version ID; use "
                f"model:{bare_id} for '{model_name}' or version:{bare_id} for "
                f"the version belonging to '{version_parent}'"
            )
        if version_data is not None:
            return self._resolve_version_reference(reference, version_data)
        if model_data is not None:
            return self._resolve_model_reference(
                CivitAIReference(
                    original=reference.original,
                    kind="model",
                    model_id=bare_id,
                ),
                model_data,
            )
        raise IdentifierError(
            f"CivitAI could not find {bare_id} as a model or version ID; if it is "
            "a file ID alone, paste the complete AIR or CivitAI URL"
        )

    @staticmethod
    def _validated_filename(filename: str) -> str:
        name = unicodedata.normalize("NFC", str(filename or ""))
        if not name or name in {".", ".."}:
            raise UnsafePathError("filename is empty or reserved")
        if name != name.strip() or name.endswith("."):
            raise UnsafePathError("filename may not start/end with spaces or dots")
        if any(ord(character) < 32 or ord(character) == 127 for character in name):
            raise UnsafePathError("filename contains control characters")
        if any(character in name for character in '<>:"/\\|?*'):
            raise UnsafePathError(
                "filename contains path separators or unsafe characters"
            )
        windows_path = PureWindowsPath(name)
        if windows_path.is_absolute() or windows_path.drive:
            raise UnsafePathError("absolute filenames are not allowed")
        if name.split(".", 1)[0].upper() in _WINDOWS_RESERVED:
            raise UnsafePathError("filename is reserved on Windows")
        if len(name.encode("utf-8")) > MAX_FILENAME_BYTES:
            raise UnsafePathError("filename is too long")
        return name

    def _safe_output_path(self, filename: str) -> Path:
        name = self._validated_filename(filename)
        candidate = self.output_dir / name
        if candidate.parent.resolve() != self.output_dir:
            raise UnsafePathError("filename escapes the output directory")
        if candidate.is_symlink():
            raise UnsafePathError(f"refusing symbolic-link target: {name}")
        if os.path.lexists(candidate) and not candidate.is_file():
            raise UnsafePathError(f"output target is not a regular file: {name}")
        return candidate

    @staticmethod
    def _control_path(path: Path) -> Path:
        return Path(f"{path}{ARIA2_EXT}")

    def _remove_owned_file(self, path: Path) -> None:
        if path.parent.resolve() != self.output_dir:
            raise UnsafePathError(
                "refusing to remove a file outside the output directory"
            )
        if path.is_symlink():
            raise UnsafePathError(f"refusing to remove symbolic link: {path.name}")
        if os.path.lexists(path):
            if not path.is_file():
                raise UnsafePathError(f"refusing to remove non-file: {path.name}")
            path.unlink()

    @contextmanager
    def _download_lock(self, target: Path) -> Iterator[None]:
        lock_id = hashlib.sha256(target.name.encode("utf-8")).hexdigest()[:20]
        lock_path = self.output_dir / f".civitai-{lock_id}.lock"
        if lock_path.is_symlink():
            raise UnsafePathError("refusing symbolic-link lock file")
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(lock_path, flags, 0o600)
        except OSError as exc:
            raise UnsafePathError("could not safely open the download lock") from exc
        handle = os.fdopen(descriptor, "a+b")
        try:
            if lock_path.stat().st_size == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise ResourceBusyError(
                    f"another download is already writing {target.name}"
                ) from exc
            try:
                yield
            finally:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
                digest.update(chunk)
        return digest.hexdigest().upper()

    def _verify_file(
        self, path: Path, resource: ResolvedCivitAIResource
    ) -> FileVerification:
        if path.parent.resolve() != self.output_dir or path.is_symlink():
            raise UnsafePathError("verification target is outside the output directory")
        if not path.is_file():
            raise IntegrityError(f"expected file is missing: {path.name}")
        size_bytes = path.stat().st_size
        if size_bytes != resource.size_bytes:
            raise IntegrityError(
                f"size mismatch for {path.name}: expected {resource.size_bytes}, got {size_bytes}"
            )
        actual_sha256 = self._sha256(path)
        if actual_sha256 != resource.sha256.upper():
            raise IntegrityError(f"SHA-256 mismatch for {path.name}")
        return FileVerification(size_bytes=size_bytes, sha256=actual_sha256)

    def _unique_target(self, target: Path) -> Path:
        counter = 1
        while True:
            candidate = self._safe_output_path(
                f"{target.stem}_{counter}{target.suffix}"
            )
            if not os.path.lexists(candidate):
                return candidate
            counter += 1

    def _require_disk_space(self, required_bytes: int, operation: str) -> None:
        free_bytes = shutil.disk_usage(self.output_dir).free
        required_with_reserve = max(0, required_bytes) + DISK_RESERVE_BYTES
        if free_bytes < required_with_reserve:
            raise DownloadError(
                f"not enough disk space for {operation}: need "
                f"{required_with_reserve} bytes, have {free_bytes}"
            )

    def _resolve_download_url(self, download_url: str) -> str:
        response: requests.Response | None = None
        try:
            response = self.session.get(
                download_url,
                allow_redirects=False,
                stream=True,
                timeout=HTTP_TIMEOUT,
            )
            if response.status_code not in {301, 302, 303, 307, 308}:
                raise DownloadError(
                    f"CivitAI download endpoint returned HTTP {response.status_code}; "
                    "a signed redirect was required"
                )
            location = response.headers.get("Location")
            if not location:
                raise DownloadError("CivitAI download redirect had no Location header")
            resolved = urljoin(download_url, location)
            parsed = urlparse(resolved)
            if parsed.scheme != "https" or not parsed.hostname:
                raise DownloadError("CivitAI returned an unsafe non-HTTPS download URL")
            if any(character in resolved for character in ("\r", "\n", "\x00")):
                raise DownloadError("CivitAI returned an invalid download URL")
            redirect_query = parse_qs(parsed.query)
            if self.token and any(
                value == self.token
                for values in redirect_query.values()
                for value in values
            ):
                raise DownloadError(
                    "CivitAI token remained in the redirected download URL"
                )
            return resolved
        except requests.RequestException as exc:
            raise DownloadError(
                f"could not resolve CivitAI download redirect: {redact(exc)}"
            ) from exc
        finally:
            if response is not None:
                response.close()

    def _aria_command(
        self, staging_path: Path, resource: ResolvedCivitAIResource
    ) -> list[str]:
        return [
            "aria2c",
            "--input-file=-",
            f"--max-connection-per-server={ARIA2_CONNECTIONS}",
            f"--split={ARIA2_SPLITS}",
            "--continue=true",
            "--file-allocation=none",
            "--auto-file-renaming=false",
            "--allow-overwrite=true",
            "--check-certificate=true",
            "--check-integrity=true",
            f"--checksum=sha-256={resource.sha256.lower()}",
            "--max-tries=5",
            "--retry-wait=2",
            "--connect-timeout=15",
            "--timeout=60",
            "--summary-interval=0",
            "--show-console-readout=false",
            "--console-log-level=warn",
            "--download-result=hide",
            f"--dir={self.output_dir}",
            f"--out={staging_path.name}",
        ]

    def _run_aria2c(self, cmd: Sequence[str], source_url: str) -> AriaResult:
        """Feed the signed URL over stdin so credentials never appear in argv."""
        if any(character in source_url for character in ("\r", "\n", "\x00")):
            raise DownloadError("download URL contains control characters")
        try:
            # The argv is fixed and the signed URL is delivered through stdin.
            process = subprocess.Popen(  # nosec B603
                list(cmd),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={
                    key: value
                    for key, value in os.environ.items()
                    if key not in {"CIVITAI_TOKEN", "civitai_token"}
                },
            )
        except FileNotFoundError as exc:
            raise DependencyError("aria2c is not installed") from exc
        if process.stdin is None or process.stdout is None:
            process.kill()
            process.wait()
            raise DownloadError("could not open aria2c input/output streams")
        tail: deque[str] = deque(maxlen=12)
        try:
            process.stdin.write(source_url + "\n")
            for option_name in ("dir", "out", "checksum"):
                prefix = f"--{option_name}="
                value = next(
                    (item[len(prefix) :] for item in cmd if item.startswith(prefix)),
                    None,
                )
                if value is not None:
                    process.stdin.write(f"  {option_name}={value}\n")
            process.stdin.close()
            for line in process.stdout:
                cleaned = redact(line).strip()
                if cleaned:
                    tail.append(cleaned)
            returncode = process.wait()
        except OSError as exc:
            self._terminate_process(process)
            raise DownloadError("aria2c communication failed") from exc
        except BaseException:
            self._terminate_process(process)
            raise
        finally:
            if not getattr(process.stdin, "closed", False):
                process.stdin.close()
            process.stdout.close()
        return AriaResult(returncode, tuple(tail))

    @staticmethod
    def _terminate_process(process: subprocess.Popen[str]) -> None:
        """Bound cleanup so an interrupted downloader does not orphan aria2c."""
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    @staticmethod
    def _diagnostic_message(result: AriaResult) -> str:
        # aria2 diagnostics may echo a short-lived signed storage URL. Keep the
        # user-facing failure stable and credential-free; the stage and exit code
        # are enough to retry or report the problem.
        return f"aria2c exited {result.returncode}"

    def _extract_safetensors(self, archive_path: Path) -> tuple[Path, ...]:
        try:
            with zipfile.ZipFile(archive_path, "r") as bundle:
                infos = bundle.infolist()
                if len(infos) > MAX_ARCHIVE_MEMBERS:
                    raise ArchiveSafetyError(
                        f"archive has {len(infos)} members; limit is {MAX_ARCHIVE_MEMBERS}"
                    )
                selected = [
                    info
                    for info in infos
                    if not info.is_dir()
                    and info.filename.lower().endswith(SAFETENSORS_EXT)
                ]
                if not selected:
                    return (archive_path,)
                expanded_bytes = sum(info.file_size for info in selected)
                if expanded_bytes > MAX_ARCHIVE_EXPANDED_BYTES:
                    raise ArchiveSafetyError(
                        f"archive expands to {expanded_bytes} bytes; limit is "
                        f"{MAX_ARCHIVE_EXPANDED_BYTES}"
                    )
                for info in selected:
                    ratio = info.file_size / max(info.compress_size, 1)
                    if ratio > MAX_ARCHIVE_COMPRESSION_RATIO:
                        raise ArchiveSafetyError(
                            f"archive member compression ratio is too high: {info.filename}"
                        )
                self._require_disk_space(expanded_bytes, "archive extraction")

                with (
                    tempfile.TemporaryDirectory(
                        prefix=".civitai-extract-", dir=self.output_dir
                    ) as temp_name,
                    ExitStack() as artifact_locks,
                ):
                    temp_root = Path(temp_name)
                    staged: list[tuple[Path, Path]] = []
                    used_names: set[str] = set()
                    for info in selected:
                        basename = info.filename.replace("\\", "/").rsplit("/", 1)[-1]
                        basename = self._validated_filename(basename)
                        candidate_name = basename
                        counter = 1
                        while candidate_name.casefold() in used_names:
                            base_path = Path(basename)
                            candidate_name = (
                                f"{base_path.stem}_{counter}{base_path.suffix}"
                            )
                            counter += 1
                        used_names.add(candidate_name.casefold())
                        staged_path = temp_root / candidate_name
                        copied = 0
                        with (
                            bundle.open(info, "r") as source,
                            staged_path.open("xb") as destination,
                        ):
                            while True:
                                chunk = source.read(HASH_CHUNK_BYTES)
                                if not chunk:
                                    break
                                copied += len(chunk)
                                if copied > info.file_size:
                                    raise ArchiveSafetyError(
                                        f"archive member exceeded declared size: {info.filename}"
                                    )
                                destination.write(chunk)
                        if copied != info.file_size:
                            raise ArchiveSafetyError(
                                f"archive member size mismatch: {info.filename}"
                            )
                        desired_path = self._safe_output_path(candidate_name)
                        final_path = desired_path
                        while True:
                            artifact_locks.enter_context(
                                self._download_lock(final_path)
                            )
                            if not os.path.lexists(final_path):
                                break
                            final_path = self._unique_target(desired_path)
                        staged.append((staged_path, final_path))

                    artifacts: list[Path] = []
                    for staged_path, final_path in staged:
                        os.replace(staged_path, final_path)
                        artifacts.append(final_path)
                self._remove_owned_file(archive_path)
                return tuple(artifacts)
        except ArchiveSafetyError:
            raise
        except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
            raise ArchiveSafetyError(f"could not safely extract ZIP: {exc}") from exc

    def _download_resource(
        self,
        resource: ResolvedCivitAIResource,
        prefer_filename: str | None = None,
        force: bool = False,
    ) -> DownloadOutcome:
        target = self._safe_output_path(prefer_filename or resource.filename)
        with ExitStack() as target_locks:
            target_locks.enter_context(self._download_lock(target))
            target = self._safe_output_path(target.name)
            if target.exists():
                try:
                    verified = self._verify_file(target, resource)
                except IntegrityError:
                    if force:
                        self._remove_owned_file(target)
                    else:
                        desired_target = target
                        while True:
                            target = self._unique_target(desired_target)
                            target_locks.enter_context(self._download_lock(target))
                            if not os.path.lexists(target):
                                break
                else:
                    if not force:
                        artifacts = (
                            self._extract_safetensors(target)
                            if target.suffix.lower() == ZIP_EXT
                            else (target,)
                        )
                        return DownloadOutcome(
                            path=artifacts[-1],
                            artifacts=artifacts,
                            status="cached",
                            size_bytes=verified.size_bytes,
                            sha256=verified.sha256,
                        )
                    self._remove_owned_file(target)

            target_key = hashlib.sha256(target.name.encode("utf-8")).hexdigest()[:12]
            staging = self.output_dir / (
                f".civitai-{resource.file_id}-{target_key}.part"
            )
            control = self._control_path(staging)
            for path in (staging, control):
                if path.parent.resolve() != self.output_dir or path.is_symlink():
                    raise UnsafePathError("unsafe staging or aria2 control path")
            if force:
                self._remove_owned_file(staging)
                self._remove_owned_file(control)

            if staging.exists() and not control.exists():
                try:
                    verified = self._verify_file(staging, resource)
                except IntegrityError:
                    self._remove_owned_file(staging)
                else:
                    os.replace(staging, target)
                    artifacts = (
                        self._extract_safetensors(target)
                        if target.suffix.lower() == ZIP_EXT
                        else (target,)
                    )
                    return DownloadOutcome(
                        path=artifacts[-1],
                        artifacts=artifacts,
                        status="recovered",
                        size_bytes=verified.size_bytes,
                        sha256=verified.sha256,
                    )

            if control.exists() and not staging.exists():
                self._remove_owned_file(control)

            resuming = staging.exists() and control.exists()
            current_size = staging.stat().st_size if staging.exists() else 0
            self._require_disk_space(
                max(0, resource.size_bytes - current_size), "download"
            )
            params = {"fileId": resource.file_id, "token": self.token}
            download_url = (
                f"{CIVITAI_API_BASE}/download/models/{resource.version_id}"
                f"?{urlencode(params)}"
            )
            signed_url = self._resolve_download_url(download_url)
            result = self._run_aria2c(self._aria_command(staging, resource), signed_url)
            if result.returncode != 0:
                raise DownloadError(self._diagnostic_message(result))
            if not staging.is_file():
                raise DownloadError(
                    f"aria2c finished without the expected staging file {staging.name}"
                )
            verified = self._verify_file(staging, resource)
            os.replace(staging, target)
            artifacts = (
                self._extract_safetensors(target)
                if target.suffix.lower() == ZIP_EXT
                else (target,)
            )
            return DownloadOutcome(
                path=artifacts[-1],
                artifacts=artifacts,
                status="resumed" if resuming else "downloaded",
                size_bytes=verified.size_bytes,
                sha256=verified.sha256,
            )

    def download(
        self,
        identifier: str,
        prefer_filename: str | None = None,
        force: bool = False,
    ) -> DownloadOutcome:
        resource = self.resolve_identifier(identifier)
        self.logger.info(
            "resolve",
            model=resource.model_id,
            version=resource.version_id,
            file=resource.file_id,
            name=resource.filename,
            format=resource.file_format,
        )
        outcome = self._download_resource(resource, prefer_filename, force)
        self.logger.success(
            "ready",
            status=outcome.status,
            bytes=outcome.size_bytes,
            sha256=outcome.sha256,
            files=len(outcome.artifacts),
            path=outcome.path,
        )
        return outcome

    def download_with_aria2(
        self,
        identifier: str,
        prefer_filename: str | None,
        force: bool = False,
    ) -> tuple[bool, Path | None]:
        """Compatibility wrapper for callers of the original class API."""
        try:
            outcome = self.download(identifier, prefer_filename, force)
            return True, outcome.path
        except DownloaderError as exc:
            self.logger.error("failure", stage=exc.stage, message=exc)
            return False, None


def get_token(args_token: str | None) -> str:
    token = os.getenv("CIVITAI_TOKEN") or os.getenv("civitai_token") or args_token
    if not token:
        raise AuthenticationError(
            "no token provided; set CIVITAI_TOKEN or pass --token"
        )
    return token


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and verify one exact CivitAI model file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -m 3268303
  %(prog)s -m model:2834417
  %(prog)s -m version:3268303
  %(prog)s -m 'civitai:2834417@3268303+3152083'
  %(prog)s -m 'https://civitai.com/models/2834417?modelVersionId=3268303'
        """,
    )
    parser.add_argument(
        "-m",
        "--identifier",
        "--model-id",
        "--model",
        dest="identifier",
        required=True,
        metavar="IDENTIFIER",
        help="CivitAI model/version ID, model URL, download URL, or AIR",
    )
    parser.add_argument("-o", "--output", default=".", help="output directory")
    parser.add_argument(
        "-t",
        "--token",
        help="CivitAI API token; CIVITAI_TOKEN is safer and preferred",
    )
    parser.add_argument("--filename", help="safe filename override, without a path")
    parser.add_argument(
        "--force", action="store_true", help="replace this exact verified target"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logger = StructuredLogger()
    try:
        downloader = CivitAIDownloader(
            get_token(args.token), args.output, logger=logger
        )
        downloader.download(args.identifier, args.filename, force=args.force)
        return 0
    except DownloaderError as exc:
        logger.error("failure", stage=exc.stage, message=exc)
        return 2 if isinstance(exc, IdentifierError) else 1
    except KeyboardInterrupt:
        logger.error("failure", stage="interrupt", message="download interrupted")
        return 130
    except Exception as exc:  # noqa: BLE001 - final CLI boundary must stay one-line
        logger.error("failure", stage="unexpected", message=redact(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
