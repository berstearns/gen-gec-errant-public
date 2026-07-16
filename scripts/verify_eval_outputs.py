#!/usr/bin/env python3
"""Walk a synced eval results tree and report missing/incomplete combos.

Expected layout (on gdrive or local mirror):
    <root>/<dataset>/<model>/analysis/summary.json

Usage:
    python scripts/verify_eval_outputs.py \\
        --root PLACEHOLDER_EVAL_ROOT/eval-gec-errant-2026-05-04
"""
import argparse
import json
import sys
from pathlib import Path

DATASETS = ["efcamdat-test", "celva-sp", "kupa-keys", "fce"]
BASE_MODELS = ["gpt2-small", "gpt2-medium", "gpt2-large",
               "pythia-70m", "pythia-160m", "pythia-410m", "pythia-1b", "pythia-1.4b",
               "smollm2-135m", "smollm2-360m", "smollm2-1.7b"]
FT_MODELS = [f"ft-{m}" for m in BASE_MODELS]
ALL_MODELS = BASE_MODELS + FT_MODELS


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()

    rows = []
    missing = 0
    for ds in DATASETS:
        for m in ALL_MODELS:
            combo_dir = args.root / ds / m
            summary = combo_dir / "analysis" / "summary.json"
            if not summary.is_file():
                rows.append((ds, m, "MISSING", "-"))
                missing += 1
                continue
            try:
                data = json.loads(summary.read_text())
                # surface a few headline fields if present
                ppl = data.get("perplexity", data.get("ppl", "-"))
                n = data.get("n_sentences", data.get("n", "-"))
                rows.append((ds, m, "ok", f"ppl={ppl} n={n}"))
            except Exception as e:
                rows.append((ds, m, f"BAD_JSON: {e}", "-"))
                missing += 1

    print(f"{'dataset':<16} {'model':<22} {'status':<10} {'notes'}")
    print("-" * 80)
    for r in rows:
        print(f"{r[0]:<16} {r[1]:<22} {r[2]:<10} {r[3]}")
    print(f"\n{len(rows)} combos checked, {missing} missing/bad")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
