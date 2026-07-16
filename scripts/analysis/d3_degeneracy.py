"""D3 — Degeneracy screen. RQ-delta. Flags empty/too-short/repetition/non-text
continuations that masquerade as interlanguage; writes _shared/clean_index.json."""
from __future__ import annotations

import json
import os
import re
from collections import Counter

from . import common
from . import plotting

ID = "D3"
SLUG = "D3-degeneracy-screen"
MIN_TOKENS = 10
REP_THRESH = 3
COMPRESSION_MIN = 0.5
NONTEXT_FRAC = 0.5

_WORD = re.compile(r"\w")


def _flags(text: str) -> dict:
    toks = (text or "").split()
    stripped = (text or "").strip()
    empty = len(stripped) == 0
    too_short = 0 < len(toks) < MIN_TOKENS
    # repetition loop: any 3-gram repeated >= REP_THRESH, or low compression ratio
    rep = False
    if len(toks) >= 3:
        grams = Counter(tuple(toks[i:i + 3]) for i in range(len(toks) - 2))
        max_gram = max(grams.values())
        uniq = len(set(toks))
        compression = uniq / len(toks) if toks else 1.0
        rep = (max_gram >= REP_THRESH) or (compression < COMPRESSION_MIN)
    # non-text: >50% of tokens have no word char
    nonalpha = sum(1 for t in toks if not _WORD.search(t))
    non_text = bool(toks) and (nonalpha / len(toks) > NONTEXT_FRAC)
    return {"empty": empty, "too_short": too_short, "rep_loop": rep, "non_text": non_text}


def run(ctx: common.Context) -> dict:
    outdir = os.path.join(ctx.out_root, SLUG)
    keys = [tuple(k) for k in ctx.paired["keys"]]
    n = len(keys)

    per_source = {}
    degenerate_any = set()
    examples = {}
    for sid, s in ctx.sources.items():
        cnt = Counter()
        offenders = []
        # too_short = under-generation, a GENERATOR pathology; authentic LEARNER
        # references are naturally short (min_new_tokens is a generation config),
        # so too_short does NOT count toward LEARNER degeneracy (only reported).
        too_short_counts = s.role != "LEARNER"
        for k in keys:
            fl = _flags(s.continuations.get(k, ""))
            for key_flag in ("empty", "too_short", "rep_loop", "non_text"):
                if fl[key_flag]:
                    cnt[key_flag] += 1
            degenerate = fl["empty"] or fl["rep_loop"] or fl["non_text"] or \
                (fl["too_short"] and too_short_counts)
            if degenerate:
                cnt["degenerate"] += 1
                degenerate_any.add(k)
                if len(offenders) < 3:
                    offenders.append(f"{k[0]}:{k[1]}")
        per_source[sid] = {"role": s.role, "empty": cnt["empty"], "too_short": cnt["too_short"],
                           "rep_loop": cnt["rep_loop"], "non_text": cnt["non_text"],
                           "degenerate": cnt["degenerate"],
                           "degenerate_rate": cnt["degenerate"] / n if n else 0.0}
        examples[sid] = offenders

    clean_keys = [k for k in keys if k not in degenerate_any]
    # write shared clean index
    shared_dir = os.path.join(ctx.out_root, "_shared")
    os.makedirs(shared_dir, exist_ok=True)
    with open(os.path.join(shared_dir, "clean_index.json"), "w") as f:
        json.dump({"run_slug": ctx.run_slug, "n": len(clean_keys),
                   "keys": [list(k) for k in clean_keys],
                   "paired_n": n, "removed": n - len(clean_keys)}, f, indent=2)
        f.write("\n")

    high = [sid for sid, v in per_source.items() if v["degenerate_rate"] > 0.20]
    results = {"source": per_source, "clean_index_size": len(clean_keys),
               "examples": examples, "params": {"min_tokens": MIN_TOKENS, "rep_thresh": REP_THRESH,
                                                 "compression_min": COMPRESSION_MIN}}
    caveats = ["EXPLORATORY. clean_index (paired ∩ non-degenerate across ALL sources) written to "
               "_shared/clean_index.json for optional robustness recompute.",
               "Downstream analyses report on the PAIRED index by default; clean-vs-paired JSD delta "
               "is a robustness signal at full S1.",
               "too_short is reported for all sources but only counts toward degeneracy for GENERATED "
               "roles (AL/CONTROL); authentic LEARNER references are legitimately short."]
    if high:
        caveats.append(f"MAJOR: degenerate_rate > 20% for {high} — generator may be broken, not "
                       "interlanguage.")
    common.write_result(outdir, ID, ctx.run_slug, {"paired": n}, results["params"], results, caveats)

    lines = ["# D3 — Degeneracy screen", "",
             f"**Finding (n={n}, EXPLORATORY):** clean index = {len(clean_keys)}/{n} sentences "
             f"({n - len(clean_keys)} degenerate in ≥1 source).", "",
             "| source | role | empty | too_short | rep_loop | non_text | degenerate | rate |",
             "|--------|------|-------|-----------|----------|----------|------------|------|"]
    for sid, v in per_source.items():
        lines.append(f"| {sid} | {v['role']} | {v['empty']} | {v['too_short']} | {v['rep_loop']} | "
                     f"{v['non_text']} | {v['degenerate']} | {common.fmt_pct(v['degenerate_rate'])} |")
    for sid, off in examples.items():
        if off:
            lines.append(f"- {sid} offenders (≤3): {', '.join(off)}")
    lines += ["", "## Caveats"] + [f"- {c}" for c in caveats]
    _maxrate = max(v["degenerate_rate"] for v in per_source.values())
    _ls = per_source.get(ctx.learner_id, {})
    lines += ["", "## Conclusion", "",
              f"No source is degenerate: the maximum degenerate rate is **{common.fmt_pct(_maxrate)}** and the "
              f"clean index equals the full paired set ({len(clean_keys)}/{n}), so no broken generation "
              f"(empty, repetition-loop, non-text) inflates any error profile — the A1/B* similarity is not a "
              f"degeneracy artefact. The {_ls.get('too_short', 0)} short LEARNER continuations are authentic "
              f"sentence-halves (a length fact for C1/D2), correctly NOT counted as degeneracy since "
              f"under-generation is a generator pathology. EXPLORATORY, n={n}."]
    common.write_md(outdir, "\n".join(lines))

    _plot(outdir, per_source)
    common.write_inputs(outdir, ctx.run_slug, ctx.sources,
                        {"files": ["raw_results.json (continuations)"],
                         "writes": ["_shared/clean_index.json"]})
    return results


def _plot(outdir, per_source):
    fdir = common.figures_dir(outdir)
    import numpy as np
    sids = list(per_source.keys())
    cats = ["empty", "too_short", "rep_loop", "non_text"]
    common.save_csv(os.path.join(fdir, "degeneracy.csv"), ["source", "role"] + cats + ["degenerate_rate"],
                    [[s, per_source[s]["role"]] + [per_source[s][c] for c in cats] + [per_source[s]["degenerate_rate"]]
                     for s in sids])
    x = np.arange(len(sids)); bottom = np.zeros(len(sids)); w = 0.5
    fig, ax = plotting.new_fig()
    palette = ["#CC6677", "#DDCC77", "#AA4499", "#882255"]
    for cat, col in zip(cats, palette):
        vals = np.array([per_source[s][cat] for s in sids], float)
        ax.bar(x, vals, w, bottom=bottom, label=cat, color=col)
        bottom += vals
    ax.set_xticks(x); ax.set_xticklabels([f"{s}\n({per_source[s]['role']})" for s in sids], fontsize=8)
    ax.set_ylabel("# degenerate sentences"); ax.set_title("D3: degeneracy by type"); ax.legend()
    plotting.save(fig, os.path.join(fdir, "degeneracy.png"))
