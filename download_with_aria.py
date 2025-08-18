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
import shutil
import subprocess
import sys
import zipfile
import re
import time
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlencode, unquote

import requests

# Constants
CIVITAI_API_BASE = "https://civitai.com/api"
ARIA2_CONNECTIONS = 8
ARIA2_SPLITS = 8
PROGRESS_INTERVAL = 10
SAFETENSORS_EXT = '.safetensors'
ZIP_EXT = '.zip'
ARIA2_EXT = '.aria2'
MIN_FILE_MB = 1  # basic sanity threshold

# Status indicators for better UX
STATUS = {
    'success': '✅',
    'error': '❌',
    'warning': '⚠️',
    'info': '🔍',
    'download': '📥',
    'extract': '📦',
    'cleanup': '🗑️',
    'file': '📁'
}


class CivitAIDownloader:
    """Handles downloading and processing of CivitAI model files."""

    def __init__(self, token: str, output_dir: str = '.'):
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
        m = re.search(r'filename\*\s*=\s*([^\'";]+)\'\'([^;]+)', header_value, flags=re.IGNORECASE)
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

        m = re.search(r'filename\s*=\s*([^;]+)', header_value, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()

        return None

    def _get_filename_from_headers(self, url: str) -> Optional[str]:
        """
        Issue a lightweight request to fetch headers and derive the filename
        from Content-Disposition. Uses GET with stream=True to be broadly compatible.
        """
        headers = {}
        # Some endpoints accept the token only as query param, but including Authorization is harmless
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        try:
            with requests.get(url, headers=headers, stream=True, timeout=30, allow_redirects=True) as r:
                cd = r.headers.get("Content-Disposition") or r.headers.get("content-disposition")
                fname = self._parse_content_disposition_filename(cd) if cd else None
                if fname:
                    print(f"{STATUS['info']} Server filename (Content-Disposition): {fname}")
                else:
                    print(f"{STATUS['warning']} No filename in Content-Disposition; will fall back to defaults")
                return fname
        except requests.RequestException as e:
            print(f"{STATUS['warning']} Could not fetch headers for filename: {e}")
            return None

    # --- Metadata (kept for optional use) --------------------------------------

    def get_model_info(self, model_id: str) -> Optional[str]:
        """Fetch model metadata from CivitAI API and return the first file name."""
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        url = f"{CIVITAI_API_BASE}/v1/model-versions/{model_id}"

        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            if 'files' in data and data['files']:
                return data['files'][0].get('name')
            print(f"{STATUS['error']} No files found in model metadata")
            return None
        except requests.RequestException as e:
            print(f"{STATUS['error']} Failed to fetch model info: {e}")
            return None

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

    def extract_safetensors_from_zip(self, zip_path: Path) -> Tuple[bool, str, Optional[Path]]:
        """
        Extract and keep only safetensors files from ZIP archive.
        Returns (ok, message, last_extracted_path or None)
        """
        print(f"{STATUS['extract']} Extracting: {zip_path.name}")
        temp_dir = self.output_dir / f"temp_extract_{zip_path.stem}"
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                safetensors_in_zip = [n for n in zip_ref.namelist() if n.lower().endswith(SAFETENSORS_EXT)]
                if not safetensors_in_zip:
                    print(f"{STATUS['warning']} No safetensors files found in archive; keeping original ZIP")
                    return True, "No safetensors files in archive - keeping original ZIP", None

                temp_dir.mkdir(exist_ok=True)
                for file_name in safetensors_in_zip:
                    zip_ref.extract(file_name, temp_dir)

                moved_count = 0
                last_moved: Optional[Path] = None
                for extracted_file in temp_dir.rglob(f"*{SAFETENSORS_EXT}"):
                    dest_file = self._get_unique_filename(self.output_dir / extracted_file.name)
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
            new_path = file_path.parent / f"{file_path.stem}_{counter}{file_path.suffix}"
            if not new_path.exists():
                return new_path
            counter += 1

    def process_downloaded_file(self, file_path: Path) -> Tuple[bool, str, Optional[Path]]:
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

    def _download_with_url(self, download_url: str, prefer_filename: Optional[str]) -> Tuple[bool, Optional[Path]]:
        """
        Download file using aria2c with the given URL.
        If prefer_filename is None, attempt to fetch it from Content-Disposition header.
        Returns (ok, downloaded_path or None).
        """
        # Resolve filename
        filename = prefer_filename or self._get_filename_from_headers(download_url)
        if not filename:
            # ultimate fallback if server doesn't send it
            filename = "download.bin"

        file_path = self.output_dir / filename
        file_path = self._get_unique_filename(file_path)  # avoid accidental overwrite collisions

        # Show URL without token for debugging
        debug_url = download_url
        if self.token:
            debug_url = debug_url.replace(f"token={self.token}", "token=***")
        print(f"{STATUS['info']} Download URL: {debug_url}")
        print(f"{STATUS['info']} Expected filename: {file_path.name}")

        # Build aria2 command
        cmd = [
            'aria2c',
            f'--max-connection-per-server={ARIA2_CONNECTIONS}',
            f'--split={ARIA2_SPLITS}',
            '--continue=true',
            '--auto-file-renaming=false',
            '--allow-overwrite=true',
            f'--summary-interval={PROGRESS_INTERVAL}',
            '--console-log-level=warn',
            '--download-result=full',
            f'--dir={self.output_dir}',
            f'--out={file_path.name}',
            download_url
        ]

        print(f"{STATUS['download']} Downloading {file_path.name}")
        print(f"{STATUS['info']} Using {ARIA2_CONNECTIONS} connections")

        try:
            subprocess.run(cmd, check=True, capture_output=False)

            # Validate expected file or discover last modified in case server changed it
            print(f"{STATUS['info']} Checking for downloaded files...")
            all_files = list(self.output_dir.glob("*"))
            print(f"{STATUS['info']} Files in directory: {[f.name for f in all_files]}")

            actual = file_path if file_path.exists() else None
            if not actual:
                print(f"{STATUS['warning']} Expected file not found: {file_path.name}")
                recent_files = [f for f in all_files if f.is_file() and (time.time() - f.stat().st_mtime) < 120]
                if recent_files:
                    actual = max(recent_files, key=lambda f: f.stat().st_mtime)
                    print(f"{STATUS['info']} Using most recent file as downloaded: {actual.name}")

            if not actual:
                print(f"{STATUS['error']} Could not locate a downloaded file")
                return False, None

            is_valid, message = self.validate_file(actual)
            if not is_valid:
                print(f"{STATUS['error']} Download validation failed: {message}")
                return False, None

            print(f"{STATUS['success']} Download complete: {message}")
            return True, actual

        except subprocess.CalledProcessError as e:
            print(f"{STATUS['error']} Download failed: {e}")
            return False, None
        except FileNotFoundError:
            print(f"{STATUS['error']} aria2c not found. Please install aria2.")
            print("  Ubuntu/Debian: sudo apt-get install aria2")
            print("  macOS: brew install aria2")
            print("  Windows: Download from https://aria2.github.io/")
            return False, None

    def download_with_aria2(self, model_id: str, prefer_filename: Optional[str], force: bool = False) -> Tuple[bool, Optional[Path]]:
        """
        Try SafeTensor first, then Diffusers ZIP as fallback.
        prefer_filename: user-supplied target name (may be None).
        Returns (ok, final_path or None).
        """
        # --- Attempt 1: SafeTensor format --------------------------------------
        params = {'type': 'Model', 'format': 'SafeTensor'}
        if self.token:
            params['token'] = self.token
        safetensor_url = f"{CIVITAI_API_BASE}/download/models/{model_id}?{urlencode(params)}"

        # If user gave a filename, prefer it for the first attempt
        target_name = prefer_filename
        if target_name:
            # cleanup any incomplete prior attempt
            self.cleanup_incomplete_download(self.output_dir / target_name)

        ok, path = self._download_with_url(safetensor_url, target_name)
        if ok and path:
            ok2, msg, final_path = self.process_downloaded_file(path)
            if ok2:
                print(f"{STATUS['success']} {msg}")
                return True, final_path or path
            else:
                print(f"{STATUS['error']} Processing failed: {msg}")

        print(f"{STATUS['warning']} SafeTensor format attempt did not succeed; trying Diffusers ZIP")

        # --- Attempt 2: Diffusers format (ZIP) ---------------------------------
        params = {'type': 'Model', 'format': 'Diffusers'}
        if self.token:
            params['token'] = self.token
        zip_url = f"{CIVITAI_API_BASE}/download/models/{model_id}?{urlencode(params)}"

        # If no explicit filename, derive from headers; ensure it has .zip suffix
        header_name = self._get_filename_from_headers(zip_url)
        if header_name and not header_name.lower().endswith(ZIP_EXT):
            # force a consistent ZIP name when server doesn't hint correctly
            header_name = f"{Path(header_name).stem}_diffusers.zip"
        if not header_name and prefer_filename:
            header_name = f"{Path(prefer_filename).stem}_diffusers.zip"

        # cleanup stale
        if header_name:
            self.cleanup_incomplete_download(self.output_dir / header_name)

        ok, path = self._download_with_url(zip_url, header_name)
        if ok and path:
            ok2, msg, final_path = self.process_downloaded_file(path)
            if ok2:
                print(f"{STATUS['success']} {msg}")
                return True, final_path or path
            else:
                print(f"{STATUS['error']} ZIP processing failed: {msg}")

        print(f"{STATUS['error']} All download attempts failed")
        return False, None


def get_token(args_token: Optional[str]) -> str:
    """Retrieve CivitAI token from environment or arguments."""
    token = os.getenv('CIVITAI_TOKEN') or os.getenv('civitai_token')
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
        description='Download AI models from CivitAI with intelligent file handling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -m 123456                    # Download model to current directory
  %(prog)s -m 123456 -o ./models        # Download to specific directory
  %(prog)s -m 123456 --force            # Force re-download
  %(prog)s -m 123456 --filename custom.safetensors  # Use custom filename
        """
    )

    parser.add_argument('-m', '--model-id', required=True, help='CivitAI model version ID')
    parser.add_argument('-o', '--output', default='.', help='Output directory (default: current directory)')
    parser.add_argument('--token', help='CivitAI API token (or set CIVITAI_TOKEN env variable)')
    parser.add_argument('--filename', help='Override filename (default: taken from server headers)')
    parser.add_argument('--force', action='store_true', help='Force re-download even if valid file exists')

    args = parser.parse_args()

    try:
        token = get_token(args.token)
        downloader = CivitAIDownloader(token, args.output)

        # Prefer user-supplied name; otherwise we’ll derive from headers at download time.
        prefer_filename = args.filename
        if prefer_filename:
            print(f"{STATUS['info']} Using custom filename: {prefer_filename}")

        ok, final_path = downloader.download_with_aria2(args.model_id, prefer_filename, force=args.force)
        if ok:
            if final_path and final_path.exists():
                print(f"{STATUS['success']} Model ready at: {final_path}")
            else:
                # fallback: pick most recent .safetensors in output
                safes = sorted(downloader.output_dir.glob("*.safetensors"), key=lambda p: p.stat().st_mtime)
                if safes:
                    print(f"{STATUS['success']} Model ready at: {safes[-1]}")
                else:
                    print(f"{STATUS['success']} Download completed successfully")
        else:
            sys.exit(1)

    except KeyboardInterrupt:
        print(f"\n{STATUS['warning']} Download interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"{STATUS['error']} Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()