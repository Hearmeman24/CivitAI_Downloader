#!/usr/bin/env python3
import requests
import argparse
import os
import sys


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--models", type=str, required=True, help="Comma-separated list of CivitAI model version IDs to download")
parser.add_argument("-t", "--token", type=str, help="CivitAI API token (if not set in environment)")
args = parser.parse_args()

token = args.token if args.token else os.getenv("civitai_token")

if not token or token == "token_here":
    print("Error: no token provided. Set the 'civitai_token' environment variable or use --token.")
    sys.exit(1)

# Split the comma-separated models into a list
model_ids = [model.strip() for model in args.models.split(',')]

# Process each model
for model_id in model_ids:
    print(f"\nProcessing model ID: {model_id}")
    
    # URL of the file to download
    url = f"https://civitai.com/api/v1/model-versions/{model_id}"

    # Perform the request
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        data = response.json()
        filename = data['files'][0]['name']
        download_url = data['files'][0]['downloadUrl']

        # Use wget with the resolved token
        os.system(
            f'wget "https://civitai.com/api/download/models/{model_id}?type=Model&format=SafeTensor&token={token}" --content-disposition')
    else:
        print(f"Error: Failed to retrieve model metadata for ID {model_id}.")
        continue  # Continue with next model instead of exiting
