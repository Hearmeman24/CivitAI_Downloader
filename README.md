# CivitAI Downloader on RunPod

Before using the downloader, you'll need to generate an API token. Instructions for this can be found in the RunPod template's README file.

## Using your token
Once you generated your token
Configure your API Token
There are two methods to set up your API token:

### Method 1: RunPod Interface
Navigate to your RunPod template
Click "Edit Template"
Find the "Environment Variables" section
Modify the civitai_token variable
Save your changes

### Method 2: Network Volume
If using a network volume, create a startup script:
Navigate to the /workspace directory
Create additional_params.sh:
Add export civitai_token="your_token_here" to the file
Save the file
The variable will be set automatically on pod startup.

## Usage

### Downloading Models
Once your token is configured, navigate to the appropriate model directory (e.g., checkpoints or loras) and run:

```download.py --model <model_version_id>```

For example:
```cd /workspace/ComfyUI/models/checkpoints```
```download.py --model 351306```

This will download DreamShaperXL V2.1 Turbo DPM ++ SDE


### Alternative Download Method

If you haven't set the environment variable, you can specify the token directly:

```download.py --model <model_version_id> --token <your_token_here>```
