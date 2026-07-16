#!/usr/bin/env python3
"""Convert FCE fce_processed.jsonl to the canonical norm-*.csv schema.

Output columns: writing_id,l1,cefr_level,text

FCE = First Certificate in English = CEFR B2 by definition (Cambridge spec),
so cefr_level is hardcoded to "B2". The "text" column is the learner text
(pre-correction); the corrected_text is dropped because the pipeline runs
its own GEC pass.

Usage:
    python scripts/fce_to_csv.py \\
        --in  PLACEHOLDER_CORPORA_SRC/data/raw-data/FCE/fce_processed.jsonl \\
        --out PLACEHOLDER_CORPORA_SRC/data/derived/norm-FCE.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True, type=Path)
    p.add_argument("--out", dest="out", required=True, type=Path)
    p.add_argument("--cefr", default="B2")
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    n_in = n_out = n_empty = 0
    with args.inp.open() as fin, args.out.open("w", newline="") as fout:
        w = csv.writer(fout, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["writing_id", "l1", "cefr_level", "text"])
        for line in fin:
            n_in += 1
            row = json.loads(line)
            text = (row.get("learner_text") or "").strip()
            if not text:
                n_empty += 1
                continue
            w.writerow([
                row.get("id", ""),
                row.get("l1", ""),
                args.cefr,
                text,
            ])
            n_out += 1

    print(f"read={n_in} wrote={n_out} skipped_empty={n_empty} -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
