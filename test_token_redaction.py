#!/usr/bin/env python3
"""Verify the CivitAI token never reaches the console in plain text.

Runs offline: the network and aria2c are stubbed. Every case drives a real code
path that used to print the token, captures stdout, and asserts the secret is
absent from what a user would see.
"""
import io
import re
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import requests

import download_with_aria as dwa

SECRET = "sk-civitai-DEADBEEF0123456789"
MODEL = "sha.safetensors"


def leaked(text):
    return SECRET in text


# --- redact() itself --------------------------------------------------------

url = f"https://civitai.com/api/download/models/1?type=Model&token={SECRET}"
assert not leaked(dwa.redact(url)), "redact must mask a token query parameter"
assert "token=***" in dwa.redact(url)
assert "type=Model" in dwa.redact(url), "redact must leave the rest of the URL readable"
assert not leaked(dwa.redact(f"Authorization: Bearer {SECRET}"))
assert not leaked(dwa.redact(f"https://civitai.com/x?a=1&api_key={SECRET}&b=2"))
assert dwa.redact("nothing secret here") == "nothing secret here"
print("ok: redact masks token, api_key and Bearer forms")


# --- the redirect-resolution failure path ------------------------------------
# requests puts the full URL in its exception text, token included.

def failing_get(*a, **kw):
    raise requests.ConnectionError(
        f"HTTPSConnectionPool(host='civitai.com', port=443): "
        f"Max retries exceeded with url: /api/download/models/1?token={SECRET}"
    )


dl = dwa.CivitAIDownloader(token=SECRET, output_dir=tempfile.mkdtemp())
real_get = dwa.requests.get
dwa.requests.get = failing_get
buf = io.StringIO()
with redirect_stdout(buf):
    dl._resolve_redirect(url)
dwa.requests.get = real_get
assert "Could not resolve download URL" in buf.getvalue(), "the warning must still print"
assert not leaked(buf.getvalue()), f"token leaked from _resolve_redirect: {buf.getvalue()}"
print("ok: a failed redirect resolution prints no token")


# --- aria2c's own output -----------------------------------------------------
# aria2c echoes the request URI in warnings and in the --download-result summary.

class FakeProc:
    def __init__(self, lines):
        self.stdout = io.StringIO("".join(lines))

    def wait(self):
        return 0


aria2_output = [
    "[ERROR] CUID#7 - Download aborted. URI="
    f"https://civitai.com/api/download/models/1?type=Model&token={SECRET}\n",
    "Download Results:\n",
    "gid   |stat|avg speed  |path/URI\n",
    f"a1b2c3|ERR |       0B/s|https://civitai.com/api/download/models/1?token={SECRET}\n",
]
real_popen = dwa.subprocess.Popen
dwa.subprocess.Popen = lambda cmd, **kw: FakeProc(aria2_output)
buf = io.StringIO()
with redirect_stdout(buf):
    rc = dl._run_aria2c(["aria2c", "http://example"])
dwa.subprocess.Popen = real_popen
assert rc == 0
assert "Download aborted" in buf.getvalue(), "aria2c output must still reach the user"
assert "Download Results:" in buf.getvalue(), "the result summary must still reach the user"
assert not leaked(buf.getvalue()), f"token leaked from aria2c output: {buf.getvalue()}"
print("ok: aria2c output is streamed with the token masked")


# --- a failing download prints an exit code, not the command line ------------
# CalledProcessError stringifies the whole argv, and the argv holds the URL.

def run_failing(make_files=lambda d: None):
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        make_files(d)
        dl2 = dwa.CivitAIDownloader(token=SECRET, output_dir=str(d))
        dl2._resolve_redirect = lambda u: (u, MODEL)  # unresolved: URL keeps the token
        dl2._run_aria2c = lambda cmd: 1  # aria2c failed
        out = io.StringIO()
        with redirect_stdout(out):
            ok, path = dl2._download_with_url(
                f"https://civitai.com/api/download/models/1?token={SECRET}", MODEL
            )
        return ok, out.getvalue()


ok, text = run_failing()
assert ok is False, "a nonzero aria2c exit must fail the download"
assert "Download failed" in text
assert not leaked(text), f"token leaked from the aria2c failure path: {text}"
print("ok: an aria2c failure reports an exit code, not the token-bearing argv")


# --- download.py no longer hands the token to a shell ------------------------
# wget prints the full URL on every run, and os.system put the token in argv.

legacy = Path(__file__).with_name("download.py").read_text()
assert "os.system" not in legacy, "download.py must not shell out with the token in the command"
assert not re.search(r"wget[^\n]*token=", legacy), "download.py must not build a wget URL with the token"
assert "redact(" in legacy, "download.py must redact what it prints"
print("ok: download.py no longer passes the token through a shell command")

print("\nall token-redaction checks passed")
