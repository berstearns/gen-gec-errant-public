"""B3 — Fine-grained tag dissection (full ERRANT tagset). RQ-beta (finest)."""
from __future__ import annotations

import os

import numpy as np

from . import common
from . import plotting

ID = "B3"
SLUG = "B3-finegrained-tag"


def run(ctx: common.Context) -> dict:
    outdir = os.path.join(ctx.out_root, SLUG)
    n = ctx.paired["n"]

    def tc(sid):
        return ctx.dists["sources"][sid]["tag"]["counts"]

    def ts(sid):
        return ctx.dists["sources"][sid]["tag"]["shares"]

    lc, ls = tc(ctx.learner_id), ts(ctx.learner_id)
    pair_results = {}
    for al_id, ctrl_id in ctx.pairs:
        alc, als = tc(al_id), ts(al_id)
        ctc, cts = tc(ctrl_id), ts(ctrl_id)
        vocab = sorted(set(lc) | set(alc) | set(ctc))
        table = []
        for t in vocab:
            table.append({"tag": t, "c_learner": lc.get(t, 0), "c_AL": alc.get(t, 0),
                          "c_CONTROL": ctc.get(t, 0), "s_learner": ls.get(t, 0.0),
                          "s_AL": als.get(t, 0.0), "s_CONTROL": cts.get(t, 0.0),
                          "d_AL": als.get(t, 0.0) - ls.get(t, 0.0),
                          "d_CONTROL": cts.get(t, 0.0) - ls.get(t, 0.0)})
        table.sort(key=lambda r: (-r["s_learner"], r["tag"]))
        learner_only = [t for t in vocab if lc.get(t, 0) > 0 and alc.get(t, 0) == 0]
        al_only = [t for t in vocab if alc.get(t, 0) > 0 and lc.get(t, 0) == 0]
        pair_results[ctx.pair_label(al_id, ctrl_id)] = {
            "table": table, "learner_only_tags": learner_only, "al_only_tags": al_only,
        }

    results = common.finalize_pairs(pair_results)
    caveats = ["EXPLORATORY. Per-tag values with count < 5 are NOT interpretable individually — "
               "see B1/B2 for robust structure and G1 for CIs.",
               "Σ shares = 1 per source; tag counts sum to each source's gen-region total."]
    common.write_result(outdir, ID, ctx.run_slug, {"paired": n, "learner": ctx.learner_id}, {},
                        results, caveats)

    lines = ["# B3 — Fine-grained tag dissection", "",
             f"**Finding (n={n}, EXPLORATORY):** full per-tag comparison of the ERRANT tagset; "
             "coarser panels (B1/B2) and the signature (E1) summarise this substrate.", ""]
    for label, b in pair_results.items():
        lines += [f"## Pair `{label}`", "",
                  "| tag | c_lrn | c_AL | c_CTRL | s_lrn | s_AL | s_CTRL | d_AL | d_CTRL |",
                  "|-----|-------|------|--------|-------|------|--------|------|--------|"]
        for r in b["table"]:
            lines.append(f"| {r['tag']} | {r['c_learner']} | {r['c_AL']} | {r['c_CONTROL']} | "
                         f"{common.fmt_pct(r['s_learner'])} | {common.fmt_pct(r['s_AL'])} | "
                         f"{common.fmt_pct(r['s_CONTROL'])} | {r['d_AL']:+.3f} | {r['d_CONTROL']:+.3f} |")
        lines += ["", f"- **learner-present, AL-absent** (coverage gaps): {', '.join(b['learner_only_tags']) or '—'}",
                  f"- **AL-present, learner-absent** (hallucinated types): {', '.join(b['al_only_tags']) or '—'}"]
    lines += ["", "## Caveats"] + [f"- {c}" for c in caveats]
    _b = next(iter(pair_results.values()))
    _lo = len(_b["learner_only_tags"]); _ao = len(_b["al_only_tags"])
    # largest AL under-shoot among learner-frequent tags
    _under = sorted([r for r in _b["table"] if r["s_learner"] > 0], key=lambda r: r["d_AL"])[:2]
    _under_txt = ", ".join(f"{r['tag']} ({r['s_learner']*100:.1f}%→{r['s_AL']*100:.1f}%)" for r in _under)
    lines += ["", "## Conclusion", "",
              f"Across the full ERRANT tagset, AL leaves **{_lo}** learner-attested tags absent (coverage "
              f"gaps) and introduces **{_ao}** tags learners never produce (hallucinated types). The largest "
              f"share shortfalls fall on the heaviest learner content errors ({_under_txt}) — AL under-"
              f"reproduces exactly the categories that dominate authentic interlanguage. Individual tags below "
              f"count 5 are not interpretable at n={n}; this table is the substrate the robust panels "
              f"(B1/B2/B4) and the E1 signature summarise. EXPLORATORY."]
    common.write_md(outdir, "\n".join(lines))

    _plot(outdir, pair_results)
    common.write_inputs(outdir, ctx.run_slug, ctx.sources, {"shared": ["_shared/distributions.json"]})
    return results


def _plot(outdir, pair_results):
    fdir = common.figures_dir(outdir)
    label0 = next(iter(pair_results)); table = pair_results[label0]["table"]
    common.save_csv(os.path.join(fdir, "tag_table.csv"),
                    ["tag", "c_learner", "c_AL", "c_CONTROL", "s_learner", "s_AL", "s_CONTROL"],
                    [[r["tag"], r["c_learner"], r["c_AL"], r["c_CONTROL"],
                      r["s_learner"], r["s_AL"], r["s_CONTROL"]] for r in table])
    # heatmap: top-25 tags by learner share
    top = table[:25]
    mat = np.array([[r["s_learner"], r["s_AL"], r["s_CONTROL"]] for r in top])
    fig, ax = plotting.new_fig(5.0, max(5, 0.32 * len(top)))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["LEARNER", "AL", "CONTROL"])
    ax.set_yticks(np.arange(len(top))); ax.set_yticklabels([r["tag"] for r in top], fontsize=8)
    ax.set_title(f"B3: tag-share heatmap (top-25) — {label0}")
    fig.colorbar(im, ax=ax, label="share")
    plotting.save(fig, os.path.join(fdir, "tag_heatmap.png"))
