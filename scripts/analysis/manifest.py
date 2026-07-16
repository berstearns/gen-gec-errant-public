"""MANIFEST.json — run-level provenance spine (H1 / 40-repro-provenance-rules).

Pins every input file (path, mtime, sha256, row count), the code git sha, the
driver command, global params, instrument metadata, and a staleness check vs
the split-fix commit e26fe1d. A result without this is void.
"""
from __future__ import annotations

import json
import os
import re
import subprocess

from . import common

SPLIT_FIX_COMMIT = "e26fe1d"  # sentence-split fix; inputs must postdate it


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _commit_epoch(ref: str):
    try:
        return int(subprocess.check_output(["git", "show", "-s", "--format=%ct", ref], text=True).strip())
    except Exception:
        return None


def _parse_provenance(run_log_path: str) -> dict:
    """Pull the provenance block key:value lines from a run.log header."""
    info = {}
    if not os.path.exists(run_log_path):
        return info
    with open(run_log_path, errors="replace") as f:
        for _ in range(40):
            line = f.readline()
            if not line:
                break
            m = re.match(r"^([a-z_]+):\s*(.+?)\s*$", line)
            if m:
                info[m.group(1)] = m.group(2)
            if "==========" in line and info:
                break
    return info


def _split_asserted(run_log_path: str) -> bool:
    if not os.path.exists(run_log_path):
        return False
    with open(run_log_path, errors="replace") as f:
        head = f.read(200_000)
    return bool(re.search(r"Split\s+\d+\s+texts?\s+into\s+\d+\s+sentences", head))


def _file_record(path: str, count_rows: bool = False) -> dict:
    if not os.path.exists(path):
        return {"path": path, "present": False}
    rec = {"path": path, "present": True, "mtime": os.path.getmtime(path),
           "sha256": common.sha256_file(path), "bytes": os.path.getsize(path)}
    if count_rows:
        with open(path, errors="replace") as f:
            rec["rows"] = sum(1 for _ in f) - 1  # minus header
    return rec


def build_manifest(ctx: common.Context, driver_cmd: str, created_ts: str,
                   analyses_run: list, analyses_skipped: list, clean_n) -> dict:
    split_epoch = _commit_epoch(SPLIT_FIX_COMMIT)
    inputs = []
    instrument = {"split_sentences": True}
    for sid, s in ctx.sources.items():
        el = os.path.join(s.run_dir, "errors_long_format.tsv")
        rr = os.path.join(s.run_dir, "raw_results.json")
        rl = os.path.join(s.run_dir, "run.log")
        prov = _parse_provenance(rl)
        split_ok = _split_asserted(rl)
        if not split_ok:
            raise SystemExit(f"BLOCKER: split_sentences not asserted in {rl} (H1 non-negotiable #4).")
        el_rec = _file_record(el, count_rows=True)
        stale = None
        if split_epoch and el_rec.get("present"):
            stale = el_rec["mtime"] < split_epoch
        inputs.append({
            "role": s.role, "id": sid, "run_dir": s.run_dir,
            "files": {"errors_long": el_rec, "raw_results": _file_record(rr),
                      "run_log": _file_record(rl)},
            "provenance": prov,
            "split_sentences_asserted": split_ok,
            "stale_vs_split_fix": stale,
        })
        instrument.setdefault("gec_model", prov.get("gec_model"))
        instrument.setdefault("run_git_sha", prov.get("git_sha"))

    try:
        import errant, spacy  # noqa
        instrument["errant_version"] = getattr(errant, "__version__", "unknown")
        instrument["spacy_model"] = "en_core_web_sm"
    except Exception:
        pass

    manifest = {
        "run_slug": ctx.run_slug,
        "created_ts": created_ts,
        "code_git_sha": _git_sha(),
        "driver_cmd": driver_cmd,
        "params_global": ctx.params,
        "learner": ctx.learner_id,
        "pairs": [f"{a}:{c}" for a, c in ctx.pairs],
        "inputs": inputs,
        "paired_n": ctx.paired["n"],
        "clean_n": clean_n,
        "analyses_run": analyses_run,
        "analyses_skipped": analyses_skipped,
        "instrument": instrument,
        "staleness_guard": {"split_fix_commit": SPLIT_FIX_COMMIT,
                            "any_stale": any(i.get("stale_vs_split_fix") for i in inputs)},
        "exploratory": ctx.exploratory,
    }
    return manifest


def write_manifest(out_root: str, manifest: dict) -> None:
    with open(os.path.join(out_root, "MANIFEST.json"), "w") as f:
        f.write(json.dumps(manifest, indent=2))
        f.write("\n")
