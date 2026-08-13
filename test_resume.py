#!/usr/bin/env python3
"""Verify _download_with_url's skip / resume / re-download decision.

Runs offline: _resolve_redirect and aria2c (_run_aria2c) are stubbed so we
only assert the decision logic around the aria2 control (.aria2) file.
"""
import tempfile
from pathlib import Path

import download_with_aria as dwa

MODEL = "sha.safetensors"


def run_case(make_files, force=False):
    """Set up a temp dir via make_files(dir), run a download, return (aria2_argv or None).

    Records the state (partial present? control present?) at the moment aria2c is
    invoked, so we can assert what cleanup ran BEFORE the download.
    """
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        make_files(d)
        dl = dwa.CivitAIDownloader(token="x", output_dir=str(d))
        # stub redirect: always resolves to a stable filename, no network
        dl._resolve_redirect = lambda url: ("http://direct/file", MODEL)
        # stub aria2c: record argv + on-disk state, "complete" the download
        captured = {}

        def fake_run(cmd):
            captured["cmd"] = cmd
            captured["control_at_invoke"] = (d / (MODEL + dwa.ARIA2_EXT)).exists()
            (d / MODEL).write_bytes(b"0" * (2 * 1024 * 1024))  # 2MB > MIN_FILE_MB
            (d / (MODEL + dwa.ARIA2_EXT)).unlink(missing_ok=True)  # aria2 clears control on finish
            return 0  # aria2c exit code

        dl._run_aria2c = fake_run
        dl._download_with_url("http://civitai/redirect", MODEL, force=force)
        return captured


def out_name(cmd):
    return next(a.split("=", 1)[1] for a in cmd if a.startswith("--out="))


# Case 1: resumable partial (file + .aria2) -> aria2 runs, SAME name, control file PRESERVED into the download
def resumable(d):
    (d / MODEL).write_bytes(b"0" * 1024)          # small partial
    (d / (MODEL + dwa.ARIA2_EXT)).write_bytes(b"ctl")  # resume state present
c = run_case(resumable)
assert c.get("cmd") is not None, "resume: aria2c should run"
assert out_name(c["cmd"]) == MODEL, f"resume: must reuse exact name, got {out_name(c['cmd'])}"
assert "--continue=true" in c["cmd"]
assert c["control_at_invoke"], "resume: .aria2 control file must survive into the aria2c call"
print("ok: resumable partial resumes under the same filename, control preserved")

# Case 2: orphaned partial (file, NO .aria2) -> deleted & re-downloaded fresh
def orphaned(d):
    (d / MODEL).write_bytes(b"0" * 1024)          # partial, no control file
c = run_case(orphaned)
assert c.get("cmd") is not None and out_name(c["cmd"]) == MODEL
print("ok: orphaned partial re-downloads")

# Case 3: complete valid file (no .aria2, >=1MB) -> skipped entirely, aria2c never runs
def complete(d):
    (d / MODEL).write_bytes(b"0" * (2 * 1024 * 1024))
c = run_case(complete)
assert c.get("cmd") is None, "complete: aria2c must NOT run for an already-valid file"
print("ok: complete file is skipped")

# Case 4: --force with a stale partial + .aria2 -> control WIPED before download (no accidental resume)
def force_stale(d):
    (d / MODEL).write_bytes(b"0" * 1024)
    (d / (MODEL + dwa.ARIA2_EXT)).write_bytes(b"ctl")
c = run_case(force_stale, force=True)
assert c.get("cmd") is not None, "force: aria2c should run"
assert not c["control_at_invoke"], "force: stale .aria2 must be deleted before download"
print("ok: --force wipes stale resume state")

# Case 5: orphaned control file (control, NO data file) -> control removed, fresh download
def orphaned_control(d):
    (d / (MODEL + dwa.ARIA2_EXT)).write_bytes(b"ctl")  # control with no data file
c = run_case(orphaned_control)
assert c.get("cmd") is not None, "orphaned-control: aria2c should run"
assert not c["control_at_invoke"], "orphaned-control: stale .aria2 must be removed"
print("ok: orphaned control file is cleaned up")

print("\nall resume-logic checks passed")
