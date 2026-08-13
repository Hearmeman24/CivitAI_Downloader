#!/usr/bin/env python3
import requests
import argparse
import os
import re
import sys

# The token travels in the download URL, so anything printed that might quote
# that URL is masked first. Duplicated from download_with_aria.py on purpose:
# both files are distributed as standalone single-file scripts.
_TOKEN_RE = re.compile(
    r"((?:[?&](?:token|api_key)=)|(?:Bearer\s+))([^&\s\"'\\]+)", re.IGNORECASE
)


def redact(value):
    """Mask API tokens in anything about to be printed."""
    return _TOKEN_RE.sub(lambda m: f"{m.group(1)}***", str(value))


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, required=True, help="CivitAI model ID to download")
parser.add_argument("-t", "--token", type=str, help="CivitAI API token (if not set in environment)")
parser.add_argument("-o", "--output", type=str, default=".", help="Output directory (default: current directory)")
args = parser.parse_args()

# Determine the token
token = os.getenv("civitai_token", args.token)
if not token:
    print("Error: no token provided. Set the 'civitai_token' environment variable or use --token.")
    sys.exit(1)

# Create output directory if it doesn't exist
output_dir = args.output
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# URL of the file to download
url = f"https://civitai.com/api/v1/model-versions/{args.model}"

# Perform the request
response = requests.get(url, stream=True)
if response.status_code == 200:
    data = response.json()
    filename = data['files'][0]['name']

    # Change to output directory before downloading
    original_dir = os.getcwd()
    os.chdir(output_dir)

    # Stream the file with requests instead of shelling out to wget: wget echoes
    # the full URL, token and all, to the terminal on every single run, and the
    # token also sat in the shell command line for anyone reading `ps`.
    download_url = (
        f"https://civitai.com/api/download/models/{args.model}"
        f"?type=Model&format=SafeTensor&token={token}"
    )

    try:
        with requests.get(download_url, stream=True, timeout=60) as download:
            download.raise_for_status()
            with open(filename, "wb") as out:
                for chunk in download.iter_content(chunk_size=1024 * 1024):
                    out.write(chunk)

        print(f"Successfully downloaded model {args.model} to {output_dir}")
    except requests.RequestException as e:
        print(f"Error: download failed: {redact(e)}")
        sys.exit(1)
    finally:
        # Change back to original directory
        os.chdir(original_dir)
else:
    print(f"Error: Failed to retrieve model metadata. Status code: {response.status_code}")
    sys.exit(1)