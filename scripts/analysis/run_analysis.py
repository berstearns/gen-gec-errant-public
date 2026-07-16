"""H1 driver — runs A1..G1 on a run-slug and writes the OUTPUT-CONTRACT tree.

Same code runs the tiny sample now and full S1 later: only --run-slug, --sources
and --pairs change. Order (dependencies): build _shared -> D1 fail-fast ->
D3 (clean_index) -> the independent panels -> G1 (needs point estimates) ->
MANIFEST -> REPORT.

  python -m scripts.analysis.run_analysis \
      --run-slug tiny-sample-2026-07-06 \
      --sources learner_baseline=outputs/s1-pilot/gpt2-small \
                gpt2-small=outputs/s1-pilot/gpt2-small \
                ft-gpt2-small=outputs/s1-pilot/ft-gpt2-small \
      --pairs ft-gpt2-small:gpt2-small \
      --learner learner_baseline \
      --analyses A1,A2,B1,B2,B3,B4,C1,D1,D2,D3,D4,E1,F1,G1 \
      --out analysis-outputs/tiny-sample-2026-07-06
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone

from . import common
from . import shared_build
from . import manifest as manifest_mod
from . import report as report_mod
from . import (a1_distributional_similarity as a1, a2_rank_correlation as a2,
               b1_operation_mru as b1, b2_pos_family as b2, b3_finegrained_tag as b3,
               b4_acquisition_categories as b4, c1_magnitude_density as c1,
               d1_region as d1, d2_length_fluency as d2, d3_degeneracy as d3,
               d4_gec_artifact as d4, e1_overrepresentation as e1,
               f1_qualitative as f1, g1_robustness as g1,
               z1_findings_synthesis as z1)

MODULES = {"A1": a1, "A2": a2, "B1": b1, "B2": b2, "B3": b3, "B4": b4, "C1": c1,
           "D2": d2, "D4": d4, "E1": e1, "F1": f1}  # independent panels
# D1, D3, G1 have special ordering/handling below.


def _roles(learner_id, pairs):
    roles = {learner_id: "LEARNER"}
    for al, ctrl in pairs:
        roles[al] = "AL"
        roles[ctrl] = "CONTROL"
    return roles


def build_context(args) -> common.Context:
    src_dirs = {}
    for tok in args.sources:
        sid, _, rundir = tok.partition("=")
        src_dirs[sid] = rundir
    pairs = []
    for p in args.pairs:
        al, _, ctrl = p.partition(":")
        pairs.append((al, ctrl))
    roles = _roles(args.learner, pairs)
    # order sources deterministically: learner, then each pair's AL, CONTROL
    ordered_ids = [args.learner] + [x for pr in pairs for x in pr]
    seen = set(); ordered = []
    for sid in ordered_ids:
        if sid not in seen and sid in src_dirs:
            ordered.append(sid); seen.add(sid)
    sources = {sid: common.load_source(sid, roles[sid], src_dirs[sid]) for sid in ordered}
    params = {"smoothing_eps": 1e-9, "jsd_base": 2, "B": args.bootstrap,
              "seed": args.seed, "ci": 0.95, "k": [5, 10], "n_per_bucket": 5}
    return common.Context(run_slug=args.run_slug, sources=sources, learner_id=args.learner,
                          pairs=pairs, paired_keys=[], paired={}, dists={},
                          out_root=args.out, params=params, exploratory=not args.confirmatory)


def run_all(ctx: common.Context, analyses: list, driver_cmd: str, created_ts: str) -> dict:
    os.makedirs(ctx.out_root, exist_ok=True)
    # 1. shared precomputations
    paired, dists = shared_build.build_shared(ctx.sources, ctx.run_slug, ctx.out_root)
    ctx.paired = paired
    ctx.dists = dists
    ctx.paired_keys = [tuple(k) for k in paired["keys"]]

    ran, skipped = [], []
    clean_n = None

    def _skip_stub(aid, slug, reason):
        d = os.path.join(ctx.out_root, slug)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "result.json"), "w") as f:
            json.dump({"analysis_id": aid, "run_slug": ctx.run_slug,
                       "status": "skipped", "reason": reason}, f, indent=2)
            f.write("\n")
        skipped.append({"id": aid, "reason": reason})

    # 2. D1 validity gate. The FAIL-FAST BLOCKER is boundary-integrity (region
    #    span alignment + count reconciliation). Prompt-drift TVD is advisory only
    #    (joint-correction, expected > 0) and does NOT gate or provision anything.
    if "D1" in analyses:
        try:
            d1_res = d1.run(ctx, fail_fast=True)
            ran.append("D1")
            drift = d1_res["prompt_drift"]["max_pairwise_tvd"]
            print(f"[D1] boundary-integrity PASS; prompt-drift TVD={drift:.3f} (advisory, "
                  "joint-correction — gen-region analyses NOT provisional).", file=sys.stderr)
        except d1.BlockerError as e:
            print(f"[D1 BLOCKER] {e} — halting (region split is broken).", file=sys.stderr)
            raise

    # 3. D3 degeneracy -> clean_index
    if "D3" in analyses:
        res = d3.run(ctx)
        clean_n = res["clean_index_size"]
        ran.append("D3")

    # 4. independent panels
    for aid in ["A1", "A2", "B1", "B2", "B3", "B4", "C1", "D2", "D4", "E1", "F1"]:
        if aid in analyses:
            MODULES[aid].run(ctx)
            ran.append(aid)

    # 5. G1 (needs the point estimates; recomputes them from the same substrate)
    if "G1" in analyses:
        g1.run(ctx)
        ran.append("G1")

    # 6. Z1 findings synthesis — runs LAST, consumes every other result.json
    if "Z1" in analyses:
        z1.run(ctx)
        ran.append("Z1")

    for aid in analyses:
        if aid not in ran:
            _skip_stub(aid, f"{aid}-unknown", "analysis id not recognised")

    # 7. MANIFEST + REPORT (REPORT leads with Z1 findings)
    manifest = manifest_mod.build_manifest(ctx, driver_cmd, created_ts, ran, skipped, clean_n)
    manifest_mod.write_manifest(ctx.out_root, manifest)
    report_mod.write_report(ctx.out_root, report_mod.assemble(ctx, manifest))
    return {"ran": ran, "skipped": skipped, "manifest": manifest}


def _result_hashes(out_root):
    """sha256 of every result.json + _shared json (for --verify determinism)."""
    hashes = {}
    for root, _, files in os.walk(out_root):
        for fn in files:
            if fn in ("result.json",) or (root.endswith("_shared") and fn.endswith(".json")):
                p = os.path.join(root, fn)
                rel = os.path.relpath(p, out_root)
                hashes[rel] = common.sha256_file(p)
    return hashes


def main(argv=None):
    ap = argparse.ArgumentParser(description="Error-dynamics analysis driver (H1)")
    ap.add_argument("--run-slug", required=True)
    ap.add_argument("--sources", nargs="+", required=True, help="id=run-dir triples")
    ap.add_argument("--pairs", nargs="+", required=True, help="AL:CONTROL (repeatable)")
    ap.add_argument("--learner", required=True)
    ap.add_argument("--analyses", default="A1,A2,B1,B2,B3,B4,C1,D1,D2,D3,D4,E1,F1,G1,Z1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--created-ts", default=None, help="MANIFEST timestamp (default: now, UTC)")
    ap.add_argument("--confirmatory", action="store_true", help="unset EXPLORATORY flag (S1 only)")
    ap.add_argument("--verify", action="store_true",
                    help="run twice into a temp tree and assert identical result.json hashes")
    args = ap.parse_args(argv)

    analyses = [a.strip() for a in args.analyses.split(",") if a.strip()]
    created_ts = args.created_ts or datetime.now(timezone.utc).isoformat()
    driver_cmd = "python -m scripts.analysis.run_analysis " + " ".join(
        [f"--run-slug {args.run_slug}", "--sources " + " ".join(args.sources),
         "--pairs " + " ".join(args.pairs), f"--learner {args.learner}",
         f"--analyses {','.join(analyses)}", f"--out {args.out}"])

    ctx = build_context(args)
    summary = run_all(ctx, analyses, driver_cmd, created_ts)
    print(f"[done] ran={summary['ran']} skipped={[s['id'] for s in summary['skipped']]} "
          f"paired_n={ctx.paired['n']} -> {args.out}")

    if args.verify:
        tmp = args.out.rstrip("/") + ".verify"
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        ctx2 = build_context(args)
        ctx2.out_root = tmp
        run_all(ctx2, analyses, driver_cmd, created_ts)
        h1 = _result_hashes(args.out)
        h2 = _result_hashes(tmp)
        diffs = [k for k in h1 if h1.get(k) != h2.get(k)] + [k for k in h2 if k not in h1]
        shutil.rmtree(tmp)
        if diffs:
            print(f"[VERIFY FAIL] {len(diffs)} result.json differ across re-runs: {diffs[:8]}",
                  file=sys.stderr)
            sys.exit(1)
        print(f"[VERIFY OK] {len(h1)} result.json + _shared json bit-identical across two runs.")


if __name__ == "__main__":
    main()
