Guide: Using the CivitAI Downloader on RunPod

Step 1: Generate an API Token

Before using the downloader, you need to generate an API token. Follow the instructions provided in the README file included in the RunPod template to obtain your token.

Step 2: Set the Environment Variable

To ensure the API token is accessible, you must set it as an environment variable in RunPod.

Method 1: Using RunPod's Interface

Navigate to your RunPod template.

Click Edit Template.

Find the Environment Variables section.

Add or edit the variable:

civitai_token="your_token_here"

Save the changes.

Method 2: Using a Network Volume

If you are using a network volume, you can create a script to set the environment variable each time the pod starts:

Navigate to the /workspace directory.

Create a file named additional_params.sh.

Add the following line to the file:

export civitai_token="your_token_here"

Save the file. This ensures the variable is set automatically upon startup.

Step 3: Download a Model

Once the environment variable is set, navigate to the appropriate model directory (e.g., checkpoints or loras) and use the following command:

download.py --model <model_version_id>

Alternative Method (Without Environment Variable)

If you did not set the environment variable, you can specify the token directly in the command:

download.py --model <model_version_id> --token <your_token_here>

This should allow you to download models efficiently on RunPod. Happy downloading!

