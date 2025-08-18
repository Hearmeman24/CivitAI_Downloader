#!/usr/bin/env python3
"""
CivitAI Model Downloader - Downloads AI models from CivitAI with intelligent file handling.
Supports automatic ZIP extraction, safetensors filtering, and robust error recovery.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlencode

import requests

# Constants
CIVITAI_API_BASE = "https://civitai.com/api"
ARIA2_CONNECTIONS = 8
ARIA2_SPLITS = 8
PROGRESS_INTERVAL = 10
SAFETENSORS_EXT = '.safetensors'
ZIP_EXT = '.zip'
ARIA2_EXT = '.aria2'

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
        self.token = token
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_model_info(self, model_id: str) -> Optional[str]:
        """Fetch model metadata from CivitAI API."""
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

    def validate_file(self, file_path: Path) -> Tuple[bool, str]:
        """Validate file existence and check for incomplete downloads."""
        if not file_path.exists():
            return False, "File does not exist"

        # Check for incomplete download markers
        if file_path.with_suffix(file_path.suffix + ARIA2_EXT).exists():
            return False, "Incomplete download detected (aria2 control file exists)"

        file_size_mb = file_path.stat().st_size / (1024 * 1024)

        # Basic size sanity check - most LoRA files are at least 1MB
        if file_size_mb < 1:
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

    def extract_safetensors_from_zip(self, zip_path: Path) -> Tuple[bool, str]:
        """Extract and keep only safetensors files from ZIP archive."""
        print(f"{STATUS['extract']} Extracting: {zip_path.name}")

        temp_dir = self.output_dir / f"temp_extract_{zip_path.stem}"

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Check if ZIP contains safetensors files before extracting
                safetensors_in_zip = [
                    name for name in zip_ref.namelist()
                    if name.lower().endswith(SAFETENSORS_EXT)
                ]

                if not safetensors_in_zip:
                    print(f"{STATUS['warning']} No safetensors files found in archive")
                    return True, "No safetensors files in archive - keeping original ZIP"

                # Extract only safetensors files
                temp_dir.mkdir(exist_ok=True)
                for file_name in safetensors_in_zip:
                    zip_ref.extract(file_name, temp_dir)

                # Move safetensors files to output directory
                moved_count = 0
                for extracted_file in temp_dir.rglob(f"*{SAFETENSORS_EXT}"):
                    dest_file = self.output_dir / extracted_file.name

                    # Handle naming conflicts
                    dest_file = self._get_unique_filename(dest_file)

                    shutil.move(str(extracted_file), str(dest_file))
                    print(f"{STATUS['file']} Extracted: {dest_file.name}")
                    moved_count += 1

                # Clean up
                print(f"{STATUS['cleanup']} Removing temporary files and original ZIP")
                shutil.rmtree(temp_dir)
                zip_path.unlink()

                return True, f"Extracted {moved_count} safetensors file(s)"

        except zipfile.BadZipFile:
            return False, "Corrupted or invalid ZIP file"
        except Exception as e:
            # Clean up on error
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            return False, f"Extraction error: {e}"

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

    def process_downloaded_file(self, file_path: Path) -> Tuple[bool, str]:
        """Process downloaded file based on its type."""
        if not file_path.exists():
            return False, "Downloaded file not found"

        if file_path.suffix.lower() == ZIP_EXT:
            return self.extract_safetensors_from_zip(file_path)
        elif file_path.suffix.lower() == SAFETENSORS_EXT:
            print(f"{STATUS['success']} File is already in safetensors format")
            return True, "File ready to use"
        else:
            print(f"{STATUS['info']} File type: {file_path.suffix}")
            return True, f"File downloaded successfully"

    def _download_with_url(self, download_url: str, filename: str) -> bool:
        """Download file using aria2c with the given URL."""
        file_path = self.output_dir / filename

        # Aria2 command with optimized settings
        cmd = [
            'aria2c',
            f'--max-connection-per-server={ARIA2_CONNECTIONS}',
            f'--split={ARIA2_SPLITS}',
            '--continue=true',
            '--auto-file-renaming=false',
            '--allow-overwrite=true',
            f'--summary-interval={PROGRESS_INTERVAL}',
            '--console-log-level=warn',  # Reduce noise
            '--download-result=full',  # Show complete results
            f'--dir={self.output_dir}',
            f'--out={filename}',
            download_url
        ]

        print(f"{STATUS['download']} Downloading {filename}")
        print(f"{STATUS['info']} Using {ARIA2_CONNECTIONS} connections")

        try:
            result = subprocess.run(cmd, check=True, capture_output=False)

            # Validate download
            is_valid, message = self.validate_file(file_path)
            if not is_valid:
                print(f"{STATUS['error']} Download validation failed: {message}")
                return False

            print(f"{STATUS['success']} Download complete: {message}")
            return True

        except subprocess.CalledProcessError as e:
            print(f"{STATUS['error']} Download failed: {e}")
            return False
        except FileNotFoundError:
            print(f"{STATUS['error']} aria2c not found. Please install aria2.")
            print("  Ubuntu/Debian: sudo apt-get install aria2")
            print("  macOS: brew install aria2")
            print("  Windows: Download from https://aria2.github.io/")
            return False

    def download_with_aria2(self, model_id: str, filename: str, force: bool = False) -> bool:
        """Download file using aria2c with multi-step fallback strategy."""
        file_path = self.output_dir / filename

        # Step 1: Check existing file (original logic)
        if not force:
            is_valid, message = self.validate_file(file_path)
            if is_valid:
                print(f"{STATUS['success']} {filename} already exists and is valid")
                print(f"   {message}")
                return True
            elif file_path.exists():
                print(f"{STATUS['info']} File validation: {message}")
                self.cleanup_incomplete_download(file_path)
        else:
            self.cleanup_incomplete_download(file_path)

        # Step 2: Check if the original filename is .safetensors - if so, use original API call
        if filename.lower().endswith(SAFETENSORS_EXT):
            print(f"{STATUS['info']} Original file is safetensors format, proceeding with standard download")

            # Build original download URL with proper encoding (original API format)
            params = {
                'type': 'Model',
                'format': 'SafeTensor',
                'token': self.token
            }
            download_url = f"{CIVITAI_API_BASE}/download/models/{model_id}?{urlencode(params)}"

            success = self._download_with_url(download_url, filename)
            if success:
                success, process_msg = self.process_downloaded_file(file_path)
                if success:
                    print(f"{STATUS['success']} {process_msg}")
                    return True
                else:
                    print(f"{STATUS['error']} Processing failed: {process_msg}")

        # Step 3: If original filename is NOT .safetensors, try SafeTensor format download
        if not filename.lower().endswith(SAFETENSORS_EXT):
            print(f"{STATUS['info']} Original file is not safetensors format, attempting SafeTensor format download")

            # Generate safetensors filename
            safetensors_filename = f"{Path(filename).stem}.safetensors"
            safetensors_path = self.output_dir / safetensors_filename

            # Clean up any existing file
            self.cleanup_incomplete_download(safetensors_path)

            # Build SafeTensor download URL
            params = {
                'type': 'Model',
                'format': 'SafeTensor',
                'token': self.token
            }
            download_url = f"{CIVITAI_API_BASE}/download/models/{model_id}?{urlencode(params)}"

            success = self._download_with_url(download_url, safetensors_filename)
            if success:
                success, process_msg = self.process_downloaded_file(safetensors_path)
                if success:
                    print(f"{STATUS['success']} {process_msg}")
                    return True
                else:
                    print(f"{STATUS['error']} Processing failed: {process_msg}")
            else:
                print(f"{STATUS['warning']} SafeTensor format download failed")

        # Step 4: Try Diffusers format download as final fallback
        print(f"{STATUS['info']} Attempting Diffusers format download (ZIP archive)")

        # Generate ZIP filename
        zip_filename = f"{Path(filename).stem}_diffusers.zip"
        zip_path = self.output_dir / zip_filename

        # Clean up any existing file
        self.cleanup_incomplete_download(zip_path)

        # Build Diffusers download URL
        params = {
            'type': 'Model',
            'format': 'Diffusers',
            'token': self.token
        }
        download_url = f"{CIVITAI_API_BASE}/download/models/{model_id}?{urlencode(params)}"

        success = self._download_with_url(download_url, zip_filename)
        if success:
            # Process the ZIP file to extract safetensors
            success, process_msg = self.process_downloaded_file(zip_path)
            if success:
                print(f"{STATUS['success']} {process_msg}")
                return True
            else:
                print(f"{STATUS['error']} ZIP processing failed: {process_msg}")

        print(f"{STATUS['error']} All download attempts failed")
        return False


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

    parser.add_argument(
        '-m', '--model-id',
        required=True,
        help='CivitAI model version ID'
    )
    parser.add_argument(
        '-o', '--output',
        default='.',
        help='Output directory (default: current directory)'
    )
    parser.add_argument(
        '--token',
        help='CivitAI API token (or set CIVITAI_TOKEN env variable)'
    )
    parser.add_argument(
        '--filename',
        help='Override filename (default: use API-provided name)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if valid file exists'
    )

    args = parser.parse_args()

    try:
        # Initialize downloader
        token = get_token(args.token)
        downloader = CivitAIDownloader(token, args.output)

        # Get filename
        if args.filename:
            filename = args.filename
            print(f"{STATUS['info']} Using custom filename: {filename}")
        else:
            filename = downloader.get_model_info(args.model_id)
            if not filename:
                print(f"{STATUS['error']} Could not determine filename from API")
                sys.exit(1)
            print(f"{STATUS['info']} Model filename: {filename}")

        # Execute download
        success = downloader.download_with_aria2(
            args.model_id,
            filename,
            force=args.force
        )

        if success:
            # Look for safetensors files in the output directory
            safetensors_files = list(downloader.output_dir.glob("*.safetensors"))
            if safetensors_files:
                print(f"{STATUS['success']} Model ready at: {safetensors_files[-1]}")
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