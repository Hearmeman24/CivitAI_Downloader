# CivitAI Model Downloader

A local command-line downloader for CivitAI LoRAs, checkpoints, embeddings, and
other model files. It resolves flexible CivitAI identifiers, selects one exact
file, downloads it with aria2, verifies CivitAI's published SHA-256 and size, and
only then makes the result visible under its final filename.

## What it provides

- Model IDs, model-version IDs, AIR strings, model-page URLs, and download URLs
- Exact `+file-id` selection from current CivitAI AIR identifiers
- SHA-256 and exact-size verification for cached and downloaded files
- Resumable aria2 transfers into private `.part` staging files
- Atomic promotion to the final filename after verification
- Bounded retries for rate limits, transient server errors, and connection failures
- Output-directory confinement, portable filename validation, and symlink refusal
- Per-filename process locks for concurrent invocations
- Bounded ZIP extraction that keeps only `.safetensors` members
- Two concise structured log lines for a normal successful download

`download_with_aria.py` is the canonical implementation. `download.py` is only a
compatibility entry point and delegates to the same implementation.

## Requirements

- Python 3.10 or newer
- [aria2](https://aria2.github.io/)
- Python packages from `requirements.txt`

Install aria2:

```bash
# Ubuntu/Debian
sudo apt-get install aria2

# macOS
brew install aria2
```

Install the Python dependency:

```bash
python3 -m pip install --requirement requirements.txt
```

To download only the standalone canonical script, use GitHub's raw endpoint:

```bash
curl --fail --location \
  --output download_with_aria.py \
  https://raw.githubusercontent.com/Hearmeman24/CivitAI_Downloader/main/download_with_aria.py
chmod +x download_with_aria.py
python3 -m pip install 'requests==2.34.2'
```

## Authentication

Create a CivitAI API token in your CivitAI account settings and expose it only to
the downloader process:

```bash
export CIVITAI_TOKEN="your-token"
```

The backward-compatible `--token` option is available, but the environment
variable is preferred because command-line arguments may be visible in shell
history and process listings. The CivitAI token is never forwarded to aria2c;
only CivitAI's short-lived signed storage URL is sent to aria2 over standard input.

## Usage

All identifier forms use the same argument:

```bash
# Bare model or version ID, auto-detected
./download_with_aria.py -m 3268303

# Explicit disambiguation
./download_with_aria.py -m model:2834417
./download_with_aria.py -m version:3268303

# Exact AIR file
./download_with_aria.py -m 'civitai:2834417@3268303+3152083'
./download_with_aria.py -m 'urn:air:minimaxh3:lora:civitai:2834417@3268303+3152083'

# Copied CivitAI URLs
./download_with_aria.py -m \
  'https://civitai.com/models/2834417?modelVersionId=3268303'
./download_with_aria.py -m \
  'https://civitai.com/api/download/models/3268303?fileId=3152083'
```

A bare number is checked as both a model and a model-version ID. If it exists in
both namespaces, the downloader stops and asks for `model:<id>` or `version:<id>`
instead of guessing. CivitAI does not expose a public reverse lookup from a bare
file ID to its version, so a file ID by itself requires the complete AIR or URL.

Other options:

```text
-o, --output DIRECTORY   Destination directory; defaults to the current directory
--filename NAME          Portable filename override; paths are rejected
--force                  Re-download and replace only the confined exact target
--token TOKEN            Backward-compatible token input; environment is safer
```

## Concise structured logs

A normal successful download emits exactly two lines:

```text
INFO resolve model=2834417 version=3268303 file=3152083 name=HMNSFW-AIO-V2.5.safetensors format=SafeTensor
OK ready status=downloaded bytes=86040232 sha256=A07732A84FD733085EB5D910F602F918FA7A3658117116927E4329F5951A9D2D files=1 path=/models/HMNSFW-AIO-V2.5.safetensors
```

Cached and resumed downloads use `status=cached` or `status=resumed`. A failure
adds one terminal `ERROR failure stage=... message=...` line; aria2 progress and
multi-line exception dumps are intentionally suppressed.

## Safety and recovery behavior

- A file is ready only when its exact byte count and SHA-256 match CivitAI metadata.
- A wrong pre-existing file is preserved and the verified download receives a
  unique filename. `--force` replaces only the validated target inside the output
  directory.
- Interrupted downloads remain in a deterministic hidden `.part` file with aria2's
  `.aria2` control file and resume on the next identical invocation.
- Concurrent attempts to write the same filename fail clearly instead of sharing a
  partial file.
- ZIP archives are inspected before extraction. Member count, expanded size,
  compression ratio, output paths, and available disk space are bounded.
- Files other than ZIP and Safetensors are downloaded but never executed. Treat
  pickle-based model formats as untrusted input and prefer Safetensors where possible.

The tool is a local, single-download CLI, not a multi-tenant hosted service. Run it
as a normal user rather than root and choose an output directory with appropriate
filesystem permissions.

## Verification

Install the development tools and run the complete offline suite:

```bash
python3 -m pip install --requirement requirements-dev.txt
python3 -m unittest discover -v
python3 -m py_compile download.py download_with_aria.py test_*.py
ruff check download.py download_with_aria.py test_*.py
ruff format --check download.py download_with_aria.py test_*.py
bandit --quiet --recursive download.py download_with_aria.py
```

The live integration test downloads the exact AIR fixture into a temporary
directory and verifies its published size, SHA-256, and two-line log contract:

```bash
CIVITAI_INTEGRATION=1 CIVITAI_TOKEN="your-token" \
  python3 -m unittest -v test_integration_civitai
```

GitHub Actions runs the offline suite on Python 3.10, 3.12, and 3.14. The live test
is available through manual workflow dispatch when the repository has a
`CIVITAI_TOKEN` Actions secret.

## License

MIT
