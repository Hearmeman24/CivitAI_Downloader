
#!/usr/bin/env python3
import requests
import os
import subprocess
import argparse
import json
from urllib.parse import urlparse, parse_qs


def get_model_info(model_id, token):
    """Get model information from CivitAI API"""
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    try:
        # Get model info
        response = requests.get(f"https://civitai.com/api/v1/model-versions/{model_id}", headers=headers)
        response.raise_for_status()
        model_data = response.json()

        # Get the latest version (first in the list)
        if not model_data.get('modelVersions'):
            raise Exception(f"No versions found for model {model_id}")

        latest_version = model_data['modelVersions'][0]

        # Get download files
        files = latest_version.get('files', [])
        if not files:
            raise Exception(f"No files found for model {model_id}")

        # Find the primary model file (usually the largest or marked as primary)
        primary_file = None
        for file in files:
            if file.get('primary', False) or file['type'] == 'Model':
                primary_file = file
                break

        if not primary_file:
            primary_file = files[0]  # Fallback to first file

        return {
            'model_name': model_data['name'],
            'version_name': latest_version['name'],
            'filename': primary_file['name'],
            'download_url': primary_file['downloadUrl'],
            'size': primary_file.get('sizeKB', 0) * 1024,
            'model_version_id': latest_version['id']  # Add this for the download URL
        }

    except requests.RequestException as e:
        raise Exception(f"Failed to fetch model info: {e}")


def download_with_aria(url, output_path, filename, token=None):
    """Download file using aria2c with proper filename"""

    # Prepare aria2c command
    cmd = [
        'aria2c',
        '--file-allocation=none',
        '--continue=true',
        '--max-connection-per-server=8',
        '--split=8',
        '--min-split-size=1M',
        '--max-concurrent-downloads=1',
        '--dir=' + output_path,
        '--out=' + filename,  # This ensures proper filename
        '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        '--content-disposition'  # Add content-disposition support
    ]

    # Add authorization header if token provided
    if token:
        cmd.extend(['--header', f'Authorization: Bearer {token}'])

    # Add URL
    cmd.append(url)

    print(f"Downloading {filename} to {output_path}")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Successfully downloaded: {filename}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Download models from CivitAI with proper filenames')
    parser.add_argument('-m', '--model-id', required=True, help='CivitAI model ID')
    parser.add_argument('-o', '--output', required=True, help='Output directory')
    parser.add_argument('--token', help='CivitAI API token')
    parser.add_argument('--filename', help='Override filename (optional)')

    args = parser.parse_args()

    try:
        # Create output directory
        os.makedirs(args.output, exist_ok=True)

        # Get model information
        print(f"Fetching info for model ID: {args.model_id}")
        model_info = get_model_info(args.model_id, args.token)

        print(f"Model: {model_info['model_name']}")
        print(f"Version: {model_info['version_name']}")
        print(f"Filename: {model_info['filename']}")
        print(f"Size: {model_info['size'] / (1024 * 1024 * 1024):.2f} GB")

        # Use custom filename if provided, otherwise use original
        filename = args.filename if args.filename else model_info['filename']

        # Construct the correct download URL
        download_url = f"https://civitai.com/api/download/models/{model_info['model_version_id']}?type=Model&format=SafeTensor"
        if args.token:
            download_url += f"&token={args.token}"

        # Download the file
        success = download_with_aria(
            download_url,
            args.output,
            filename,
            args.token
        )

        if success:
            print(f"✅ Download completed: {os.path.join(args.output, filename)}")
        else:
            print(f"❌ Download failed for model {args.model_id}")
            exit(1)

    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()