# CivitAI Model Downloader 🚀

A robust Python script for downloading AI models (LoRA, Checkpoints, Embeddings) from CivitAI with intelligent file handling, automatic ZIP extraction, and resume support.

## ✨ Features

- **🔄 Smart Download Management** - Multi-connection downloads with aria2 for maximum speed
- **📦 Automatic ZIP Processing** - Extracts and filters `.safetensors` files automatically
- **🔧 Resume Support** - Continue interrupted downloads without starting over
- **✅ File Validation** - Detects and cleans up corrupted or incomplete downloads
- **🔐 Secure Token Handling** - Supports environment variables for API authentication
- **📊 Progress Tracking** - Real-time download progress with clear status indicators
- **🎯 Intelligent File Management** - Handles naming conflicts and cleans up temporary files
- **🧭 Flexible CivitAI Identifiers** - Accepts model IDs, version IDs, model/download URLs, and AIR strings
- **🎯 Exact File Selection** - Honors the `+file-id` in CivitAI AIR strings instead of guessing among version files

## 📋 Requirements

- Python 3.6+
- [aria2](https://aria2.github.io/) - High-speed download utility
- [requests](https://pypi.org/project/requests/) - HTTP library for API calls

## 🔧 Installation

### 1. Install aria2

**Ubuntu/Debian:**
```bash
sudo apt-get install aria2
```

**macOS:**
```bash
brew install aria2
```

**Windows:**
Download from [aria2 releases](https://github.com/aria2/aria2/releases)

### 2. Install Python dependencies

```bash
pip install requests
```

### 3. Download the script

```bash
wget https://github.com/Hearmeman24/CivitAI_Downloader/blob/main/download_with_aria.py
chmod +x download_with_aria.py
```

## 🔑 Authentication

Get your CivitAI API token from [CivitAI Account Settings](https://civitai.com/user/account).

Set it as an environment variable:

```bash
export CIVITAI_TOKEN="your_token_here"
```

Or add to your `~/.bashrc` or `~/.zshrc` for permanent use:

```bash
echo 'export CIVITAI_TOKEN="your_token_here"' >> ~/.bashrc
source ~/.bashrc
```

## 📖 Usage

### Basic Usage

Paste a model ID, model-version ID, model page URL, download URL, or complete
CivitAI AIR identifier into the same argument:

```bash
./download_with_aria.py -m 3268303
./download_with_aria.py -m model:2834417
./download_with_aria.py -m 'civitai:2834417@3268303+3152083'
```

A plain number is checked as both a model ID and a model-version ID. A model ID
downloads that model's default (first listed) published version; a version ID
downloads that version's primary file. If the number exists in both namespaces,
the downloader stops instead of risking the wrong download—prefix it with
`model:` or `version:` to choose explicitly.

The new file ID is honored when it arrives with its required context, such as
`civitai:2834417@3268303+3152083` or
`https://civitai.com/api/download/models/3268303?fileId=3152083`. CivitAI does
not expose a public file-ID-to-version reverse lookup, so a bare file ID by
itself cannot be resolved safely; paste the complete AIR or URL shown by
CivitAI.

### Advanced Options

```bash
# Download to specific directory
./download_with_aria.py -m 123456 -o ./models

# Use custom filename
./download_with_aria.py -m 123456 --filename "my_custom_model.safetensors"

# Force re-download (ignore existing files)
./download_with_aria.py -m 123456 --force

# Provide token via command line (not recommended for security)
./download_with_aria.py -m 123456 --token "your_token_here"
```

### Command Line Arguments

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--identifier` (`--model-id` alias) | `-m` | CivitAI model/version ID, model/download URL, or AIR (required) | - |
| `--output` | `-o` | Output directory | Current directory |
| `--token` | - | CivitAI API token | From environment |
| `--filename` | - | Override default filename | From API |
| `--force` | - | Force re-download | False |

## 🎯 Examples

### Download a LoRA model
```bash
# Any unambiguous plain model or version ID
./download_with_aria.py -m 3268303

# Explicit IDs (useful if a plain number is ambiguous)
./download_with_aria.py -m model:2834417
./download_with_aria.py -m version:3268303

# Exact file from the AIR shown on CivitAI
./download_with_aria.py -m 'urn:air:minimaxh3:lora:civitai:2834417@3268303+3152083'

# A copied CivitAI model page works too
./download_with_aria.py -m 'https://civitai.com/models/2834417?modelVersionId=3268303'
```

### Download multiple models to organized folders
```bash
# Download character LoRA
./download_with_aria.py -m 245589 -o ./models/lora/characters

# Download style LoRA
./download_with_aria.py -m 234567 -o ./models/lora/styles

# Download checkpoint
./download_with_aria.py -m 345678 -o ./models/checkpoints
```

### Batch download with a simple script
```bash
#!/bin/bash
# download_batch.sh

models=(245589 234567 345678 456789)
for model_id in "${models[@]}"; do
    ./download_with_aria.py -m "$model_id" -o ./models
done
```

## 🔍 How It Works

1. **Resolves the Input** - Classifies a model ID, version ID, CivitAI URL, or AIR without guessing across ambiguous ID namespaces
2. **Selects the Exact File** - Uses an AIR/URL file ID when supplied, otherwise selects the resolved version's primary file
3. **Validates Existing Files** - Checks if a valid file already exists
4. **Downloads with aria2** - Uses 8 parallel connections for speed
5. **Processes Downloaded Files**:
   - `.safetensors` - Keeps as-is
   - `.zip` - Extracts only `.safetensors` files, removes archive
   - Other formats - Keeps as downloaded
6. **Cleanup** - Removes temporary files and failed downloads

## 📊 Status Indicators

The script uses clear emoji indicators for status:

- ✅ Success - Operation completed successfully
- ❌ Error - Operation failed
- ⚠️ Warning - Important notice
- 🔍 Info - Information message
- 📥 Download - Downloading file
- 📦 Extract - Extracting archive
- 🗑️ Cleanup - Removing temporary files
- 📁 File - File operation

## 🐛 Troubleshooting

### "No CivitAI token provided"
Set your token as an environment variable or use the `--token` argument.

### "aria2c not found"
Install aria2 using the installation instructions above.

### "Download validation failed"
The file may be corrupted. Use `--force` to re-download:
```bash
./download_with_aria.py -m 123456 --force
```

### Slow downloads
CivitAI may throttle downloads. The script uses 8 connections by default for optimal speed.

### "No safetensors files found in archive"
Some models may use different formats. The original ZIP is kept in this case.

### "ID is both a model ID and a version ID"
The same number exists in two independent CivitAI namespaces. Re-run with
`model:<id>` to use the model's default version or `version:<id>` to download
that exact version.

### "CivitAI could not find ... as a model or version ID"
If you pasted only the new file ID, copy the complete AIR or download URL from
CivitAI instead. The file ID needs its version ID to select the correct file.

## ✅ Verification

The checks are offline unless a test explicitly says otherwise:

```bash
python3 test_identifier_resolution.py
python3 test_resume.py
python3 test_token_redaction.py
python3 -m py_compile download_with_aria.py download.py test_*.py
```

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Report bugs
- Suggest new features
- Submit pull requests

## 📄 License

MIT License - feel free to use this script in your projects.

## 🙏 Acknowledgments

- [CivitAI](https://civitai.com) for providing the API and hosting models
- [aria2](https://aria2.github.io/) for the excellent download utility
- The AI art community for creating and sharing models

## 📝 Notes

- Always respect model licenses and creator terms
- Be mindful of CivitAI's rate limits and terms of service
- Large checkpoint files (>5GB) may take significant time to download
- The script requires a stable internet connection for resume to work properly

---

**Need help?** Open an issue on GitHub or check CivitAI's documentation for model-specific questions.
