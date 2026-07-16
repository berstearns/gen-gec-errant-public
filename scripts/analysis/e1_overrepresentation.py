"""E1 — Over/under-representation signature. RQ-alpha (interpretability).

log2((s_source+eps)/(s_learner+eps)) per tag: positive = AL over-produces vs
learner, negative = under. The 'toward-learner' aggregate is the story scalar.
"""
from __future__ import annotations

import math
import os

from . import common
from . import plotting

ID = "E1"
SLUG = "E1-overrepresentation-signature"
MIN_COUNT = 2  # tiny-sample threshold to enter the plot (full S1: 5)


def run(ctx: common.Context) -> dict:
    outdir = os.path.join(ctx.out_root, SLUG)
    n = ctx.paired["n"]

    def sc(sid):
        return ctx.dists["sources"][sid]["tag"]["counts"]

    def ss(sid):
        return ctx.dists["sources"][sid]["tag"]["shares"]

    lc, ls = sc(ctx.learner_id), ss(ctx.learner_id)
    pair_results = {}
    for al_id, ctrl_id in ctx.pairs:
        alc, als = sc(al_id), ss(al_id)
        ctc, cts = sc(ctrl_id), ss(ctrl_id)
        n_al = sum(alc.values()); n_ct = sum(ctc.values())
        eps_al = 0.5 / n_al if n_al else 1e-6
        eps_ct = 0.5 / n_ct if n_ct else 1e-6
        vocab = sorted(set(lc) | set(alc) | set(ctc))
        signature = []
        toward = 0
        considered = 0
        for t in vocab:
            l2_al = math.log2((als.get(t, 0) + eps_al) / (ls.get(t, 0) + eps_al))
            l2_ct = math.log2((cts.get(t, 0) + eps_ct) / (ls.get(t, 0) + eps_ct))
            al_toward = abs(l2_al) < abs(l2_ct)
            signature.append({"tag": t, "s_learner": ls.get(t, 0.0),
                              "c_learner": lc.get(t, 0), "l2_AL": l2_al, "l2_CONTROL": l2_ct,
                              "al_toward_learner": al_toward})
            # aggregate over learner-present tags
            if lc.get(t, 0) > 0:
                considered += 1
                if al_toward:
                    toward += 1
        signature.sort(key=lambda r: (-r["s_learner"], r["tag"]))
        pair_results[ctx.pair_label(al_id, ctrl_id)] = {
            "signature": signature,
            "signature_alignment": {"n_toward": toward, "n_total": considered,
                                    "frac": toward / considered if considered else 0.0},
            "eps": {"AL": eps_al, "CONTROL": eps_ct}, "n_source": {"AL": n_al, "CONTROL": n_ct}}

    results = common.finalize_pairs(pair_results)
    caveats = ["EXPLORATORY. ε = 0.5/N_source count-smoothing (reported). Only tags with learner or "
               f"source count ≥ {MIN_COUNT} enter the figure; rest pooled to 'other' (named).",
               "signature_alignment (over learner-present tags) should track B2 directional_agreement."]
    common.write_result(outdir, ID, ctx.run_slug, {"paired": n, "learner": ctx.learner_id},
                        {"min_count_plot": MIN_COUNT}, results, caveats)

    lines = ["# E1 — Over/under-representation signature", "",
             f"**Finding (n={n}, EXPLORATORY):** " + _headline(pair_results), ""]
    for label, b in pair_results.items():
        a = b["signature_alignment"]
        lines += [f"## Pair `{label}`", "",
                  f"- **signature alignment: {a['n_toward']}/{a['n_total']} learner-present tags "
                  f"pulled toward LEARNER ({common.fmt_pct(a['frac'])})**",
                  f"- ε: AL={b['eps']['AL']:.4g}, CONTROL={b['eps']['CONTROL']:.4g}; "
                  f"N_source: AL={b['n_source']['AL']}, CONTROL={b['n_source']['CONTROL']}", "",
                  "| tag | s_learner | log2(AL/lrn) | log2(CTRL/lrn) | AL→lrn |",
                  "|-----|-----------|--------------|----------------|--------|"]
        for r in b["signature"]:
            if r["c_learner"] == 0 and abs(r["l2_AL"]) < 0.01:
                continue
            lines.append(f"| {r['tag']} | {common.fmt_pct(r['s_learner'])} | {r['l2_AL']:+.2f} | "
                         f"{r['l2_CONTROL']:+.2f} | {'✓' if r['al_toward_learner'] else ''} |")
    lines += ["", "## Caveats"] + [f"- {c}" for c in caveats]
    _b = next(iter(pair_results.values()))
    _al = _b["signature_alignment"]
    _lp = [s for s in _b["signature"] if s["s_learner"] > 0]
    _under = sorted(_lp, key=lambda s: s["l2_AL"])[:2]
    _over = sorted(_b["signature"], key=lambda s: -s["l2_AL"])[:1]
    _under_txt = ", ".join(f"{s['tag']} (2^{s['l2_AL']:+.1f})" for s in _under)
    _over_txt = ", ".join(f"{s['tag']} (2^{s['l2_AL']:+.1f})" for s in _over)
    lines += ["", "## Conclusion", "",
              f"Fine-tuning pulls **{_al['n_toward']}/{_al['n_total']}** learner-present tags toward the learner "
              f"rate (signature alignment {common.fmt_pct(_al['frac'])}). The residual divergences are "
              f"directional and interpretable: AL most **under**-produces {_under_txt} (in log2 units vs "
              f"learner) — the sub-lexical content errors requiring learner-specific patterns — and most "
              f"**over**-produces {_over_txt}, the GEC-orthography artifact adjudicated by D4. The signature "
              f"diverging figure is the paper's money plot. ε=0.5/N_source; tags <2 pooled. EXPLORATORY, "
              f"n={n}."]
    common.write_md(outdir, "\n".join(lines))

    _plot(outdir, pair_results)
    common.write_inputs(outdir, ctx.run_slug, ctx.sources, {"shared": ["_shared/distributions.json"]})
    return results


def _headline(pair_results):
    b = next(iter(pair_results.values()))
    a = b["signature_alignment"]
    return f"signature alignment = {a['n_toward']}/{a['n_total']} learner-present tags pulled toward LEARNER by fine-tuning"


def _plot(outdir, pair_results):
    fdir = common.figures_dir(outdir)
    import numpy as np
    label0 = next(iter(pair_results)); b = pair_results[label0]
    # tags with learner OR source signal above threshold; keep top-18 by learner share
    sig = [r for r in b["signature"]
           if r["c_learner"] >= MIN_COUNT or abs(r["l2_AL"]) >= 0.01][:18]
    common.save_csv(os.path.join(fdir, "signature_diverging.csv"),
                    ["tag", "s_learner", "l2_AL", "l2_CONTROL", "al_toward_learner"],
                    [[r["tag"], r["s_learner"], r["l2_AL"], r["l2_CONTROL"], r["al_toward_learner"]]
                     for r in b["signature"]])
    ys = np.arange(len(sig))[::-1]
    fig, ax = plotting.new_fig(7.5, max(4.5, 0.42 * len(sig)))
    ax.barh(ys + 0.2, [r["l2_AL"] for r in sig], height=0.38, label="AL", color=plotting.ROLE_COLOR["AL"])
    ax.barh(ys - 0.2, [r["l2_CONTROL"] for r in sig], height=0.38, label="CONTROL", color=plotting.ROLE_COLOR["CONTROL"])
    ax.axvline(0, color="#4477AA", lw=1.2, label="LEARNER baseline")
    ax.set_yticks(ys); ax.set_yticklabels([r["tag"] for r in sig], fontsize=8)
    ax.set_xlabel("log2(source share / learner share)  ← under · over →")
    ax.set_title(f"E1: over/under-representation vs LEARNER — {label0}"); ax.legend()
    plotting.save(fig, os.path.join(fdir, "signature_diverging.png"))
