#!/usr/bin/env python3
import requests
import os
import subprocess
import argparse
import sys
import shutil


def get_model_info(model_id, token):
    """Get model information from CivitAI API"""
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    try:
        # Get model info
        response = requests.get(f"https://civitai.com/api/v1/model-versions/{model_id}", headers=headers)
        response.raise_for_status()
        if response.status_code == 200:
            data = response.json()
            filename = data['files'][0]['name']
            return filename
        return None
    except requests.RequestException as e:
        raise Exception(f"Failed to fetch model info: {e}")


def download_with_aria(model_id, output_path, filename, token):
    """Download file using aria2c with list format"""

    # Build command as list (more reliable)
    cmd = [
        'aria2c',
        '-x', '2',
        '-s', '2',
        '--continue=true',
        '--summary-interval=10',
        f'--dir={output_path}',
        f'--out={filename}',
        f'https://civitai.com/api/download/models/{model_id}?type=Model&format=SafeTensor&token={token}'
    ]

    print(f"Downloading {filename} to {output_path}")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True)
        print(f"✅ Successfully downloaded: {filename}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e}")
        return False


def get_civitai_token(args_token):
    """Get CivitAI token from environment variable or arguments"""
    # Check environment variable first
    env_token = os.getenv('civitai_token')

    if env_token:
        print("✅ Using CIVITAI_TOKEN from environment variable")
        return env_token
    elif args_token:
        print("✅ Using token from command line argument")
        return args_token
    else:
        print("❌ Error: No CivitAI token provided.")
        print("Either set CIVITAI_TOKEN environment variable or use --token argument")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Download models from CivitAI with proper filenames')
    parser.add_argument('-m', '--model-id', required=True, help='CivitAI model ID')
    parser.add_argument('-o', '--output', default='.', help='Output directory (default: current directory)')
    parser.add_argument('--token', help='CivitAI API token (optional if CIVITAI_TOKEN env var is set)')
    parser.add_argument('--filename', help='Override filename (optional)')

    args = parser.parse_args()

    try:
        # Get token from environment or arguments
        token = get_civitai_token(args.token)

        # Get filename from API
        filename = get_model_info(args.model_id, token)

        # Use custom filename if provided
        if args.filename:
            filename = args.filename

        if not filename:
            print("❌ Could not determine filename for the model")
            sys.exit(1)

        # Create output directory
        os.makedirs(args.output, exist_ok=True)

        # Download the file
        success = download_with_aria(
            args.model_id,
            args.output,
            filename,
            token
        )

        if success:
            print(f"✅ Download completed: {os.path.join(args.output, filename)}")
        else:
            print(f"❌ Download failed for model {args.model_id}")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()