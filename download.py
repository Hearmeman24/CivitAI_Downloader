#!/usr/bin/env python3
import requests
import argparse
import os
import sys


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, required=True, help="CivitAI model version ID to download")
parser.add_argument("-t", "--token", type=str, help="CivitAI API token (if not set in environment)")
args = parser.parse_args()

token = args.token if args.token else os.getenv("civitai_token")

if not token or token == "token_here":
    print("Error: no token provided. Set the 'civitai_token' environment variable or use --token.")
    sys.exit(1)

# URL of the file to download
url = f"https://civitai.com/api/v1/model-versions/{args.model}"

# Perform the request
response = requests.get(url, stream=True)
if response.status_code == 200:
    data = response.json()
    filename = data['files'][0]['name']
    download_url = data['files'][0]['downloadUrl']

    # Use wget with the resolved token
    os.system(
        f'wget "https://civitai.com/api/download/models/{args.model}?type=Model&format=SafeTensor&token={token}" --content-disposition')
else:
    print("Error: Failed to retrieve model metadata.")
    sys.exit(1)
