
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
        response = requests.get(f"https://civitai.com/api/v1/model-versions/{model_id}")
        response.raise_for_status()
        if response.status_code == 200:
            data = response.json()
            filename = data['files'][0]['name']
            return filename
        return None
    except requests.RequestException as e:
        raise Exception(f"Failed to fetch model info: {e}")


def download_with_aria(model_id, output_path, filename, token=None):
    """Download file using aria2c with proper filename"""

    # Prepare aria2c command
    cmd = (
        f'aria2c -x 8 -s 8 --file-allocation=none --continue=true --summary-interval=10 '
        f'--dir="{output_path}" '
        f'--out="{filename}" '
        f'"https://civitai.com/api/download/models/{model_id}?type=Model&format=SafeTensor&token={token}"'
    )

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
        filename = get_model_info(args.model_id, args.token)
        # Create output directory
        os.makedirs(args.output, exist_ok=True)

        # Construct the correct download URL
        download_url = f"https://civitai.com/api/download/models/{args.model_id}?type=Model&format=SafeTensor"
        if args.token:
            download_url += f"&token={args.token}"

        # Download the file
        success = download_with_aria(
            args.model_id,
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