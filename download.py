#!/usr/bin/env python3
import requests
import argparse
import os
import sys

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
    download_url = data['files'][0]['downloadUrl']

    # Change to output directory before downloading
    original_dir = os.getcwd()
    os.chdir(output_dir)

    try:
        # Use wget with the resolved token
        exit_code = os.system(
            f'wget "https://civitai.com/api/download/models/{args.model}?type=Model&format=SafeTensor&token={token}" --content-disposition')

        if exit_code != 0:
            print(f"Error: wget failed with exit code {exit_code}")
            os.chdir(original_dir)
            sys.exit(1)

        print(f"Successfully downloaded model {args.model} to {output_dir}")
    finally:
        # Change back to original directory
        os.chdir(original_dir)
else:
    print(f"Error: Failed to retrieve model metadata. Status code: {response.status_code}")
    sys.exit(1)