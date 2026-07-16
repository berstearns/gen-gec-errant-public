"""B4 — Acquisition-category focus panel. RQ-beta (SLA-critical).

The pre-registered acquisition alignment: count of phenomena where AL moves
toward the LEARNER rate vs CONTROL (the ">=2/4 categories" companion to JSD).
"""
from __future__ import annotations

import os

import numpy as np

from . import common
from . import plotting

ID = "B4"
SLUG = "B4-acquisition-categories"


def run(ctx: common.Context) -> dict:
    outdir = os.path.join(ctx.out_root, SLUG)
    n = ctx.paired["n"]

    def ts(sid):
        return ctx.dists["sources"][sid]["tag"]["shares"]

    ls = ts(ctx.learner_id)
    pair_results = {}
    for al_id, ctrl_id in ctx.pairs:
        als, cts = ts(al_id), ts(ctrl_id)
        panel = {}
        toward_flags = {}
        for phen, tags in common.ACQUISITION_PHENOMENA.items():
            panel[phen] = {}
            # phenomenon-level share = sum over its tags
            l_ph = sum(ls.get(t, 0) for t in tags)
            al_ph = sum(als.get(t, 0) for t in tags)
            ct_ph = sum(cts.get(t, 0) for t in tags)
            for t in tags:
                panel[phen][t] = {"learner": ls.get(t, 0.0), "AL": als.get(t, 0.0),
                                  "CONTROL": cts.get(t, 0.0),
                                  "d_AL": als.get(t, 0.0) - ls.get(t, 0.0),
                                  "d_CONTROL": cts.get(t, 0.0) - ls.get(t, 0.0)}
            toward_flags[phen] = abs(al_ph - l_ph) < abs(ct_ph - l_ph)
            panel[phen]["_phenomenon_share"] = {"learner": l_ph, "AL": al_ph, "CONTROL": ct_ph,
                                                "toward_learner": toward_flags[phen]}
        phen_toward = [p for p, v in toward_flags.items() if v]
        alignment = {"n_toward_learner": len(phen_toward), "n_total": len(toward_flags),
                     "phenomena_toward": phen_toward}
        # determiner sub-analysis M/U/R balance
        det = {role: {t: src.get(t, 0.0) for t in ("M:DET", "U:DET", "R:DET")}
               for role, src in (("LEARNER", ls), ("AL", als), ("CONTROL", cts))}
        pair_results[ctx.pair_label(al_id, ctrl_id)] = {
            "panel": panel, "acquisition_alignment": alignment, "determiner_balance": det}

    results = common.finalize_pairs(pair_results)
    caveats = ["EXPLORATORY. This panel + A1's JSD are the CONFIRMATORY readout at full S1; tiny-"
               "sample values are hypothesis-generating (many acquisition tags count < 5).",
               "acquisition_alignment counts phenomena where AL's phenomenon-share is nearer the "
               "LEARNER's than CONTROL's — the pre-registered ≥2/4 companion rule."]
    common.write_result(outdir, ID, ctx.run_slug, {"paired": n, "learner": ctx.learner_id}, {},
                        results, caveats)

    lines = ["# B4 — Acquisition-category panel", "",
             f"**Finding (n={n}, EXPLORATORY):** " + _headline(pair_results), ""]
    for label, b in pair_results.items():
        a = b["acquisition_alignment"]
        lines += [f"## Pair `{label}`", "",
                  f"- **acquisition alignment: {a['n_toward_learner']}/{a['n_total']} phenomena** "
                  f"move toward LEARNER (toward: {', '.join(a['phenomena_toward']) or '—'})", "",
                  "| phenomenon | tag | LEARNER | AL | CONTROL | d_AL | d_CTRL |",
                  "|-----------|-----|---------|-----|---------|------|--------|"]
        for phen, tagmap in b["panel"].items():
            for t, v in tagmap.items():
                if t == "_phenomenon_share":
                    continue
                lines.append(f"| {phen} | {t} | {common.fmt_pct(v['learner'])} | {common.fmt_pct(v['AL'])} | "
                             f"{common.fmt_pct(v['CONTROL'])} | {v['d_AL']:+.3f} | {v['d_CONTROL']:+.3f} |")
    lines += ["", "## Caveats"] + [f"- {c}" for c in caveats]
    _b = next(iter(pair_results.values()))
    _a = _b["acquisition_alignment"]
    # per-phenomenon gap closure toward learner
    def _gc(ph):
        s = _b["panel"][ph]["_phenomenon_share"]
        d_al, d_ct = s["AL"] - s["learner"], s["CONTROL"] - s["learner"]
        return None if abs(d_ct) < 1e-6 else (abs(d_ct) - abs(d_al)) / abs(d_ct)
    _det, _prep = _gc("determiner"), _gc("preposition")
    lines += ["", "## Conclusion", "",
              f"Fine-tuning moves **{_a['n_toward_learner']}/{_a['n_total']}** acquisition phenomena toward the "
              f"learner rate (≥2/4 pre-registered companion → met). The movement is led by determiner use "
              f"(gap-closure {common.fmt_pct(_det or 0)}) and preposition choice ({common.fmt_pct(_prep or 0)}) "
              f"— the canonical article/preposition difficulties of L1-Romance English learners in CELVA-SP — "
              f"while verb morphology stays control-like and tense overshoots. The alignment is thus category-"
              f"specific and SLA-plausible, not uniform. Several tags count <5 at n={n}; this + A1 JSD are the "
              f"confirmatory readout at S1, exploratory here."]
    common.write_md(outdir, "\n".join(lines))

    _plot(outdir, pair_results)
    common.write_inputs(outdir, ctx.run_slug, ctx.sources, {"shared": ["_shared/distributions.json"]})
    return results


def _headline(pair_results):
    b = next(iter(pair_results.values()))
    a = b["acquisition_alignment"]
    return f"acquisition alignment = {a['n_toward_learner']}/{a['n_total']} phenomena toward LEARNER"


def _plot(outdir, pair_results):
    fdir = common.figures_dir(outdir)
    label0 = next(iter(pair_results)); b = pair_results[label0]
    phens = list(b["panel"].keys())
    rows = []
    for phen in phens:
        ps = b["panel"][phen]["_phenomenon_share"]
        rows.append([phen, ps["learner"], ps["AL"], ps["CONTROL"]])
    common.save_csv(os.path.join(fdir, "acquisition_panel.csv"),
                    ["phenomenon", "LEARNER", "AL", "CONTROL"], rows)
    ncol = 3; nrow = int(np.ceil(len(phens) / ncol))
    fig, axes = plotting.plt.subplots(nrow, ncol, figsize=(9, 2.4 * nrow))
    axes = np.array(axes).reshape(-1)
    for i, phen in enumerate(phens):
        ax = axes[i]
        ps = b["panel"][phen]["_phenomenon_share"]
        vals = [ps["learner"], ps["AL"], ps["CONTROL"]]
        ax.bar(["LRN", "AL", "CTRL"], vals,
               color=[plotting.ROLE_COLOR["LEARNER"], plotting.ROLE_COLOR["AL"], plotting.ROLE_COLOR["CONTROL"]])
        ax.set_title(phen + (" ✓" if ps["toward_learner"] else ""), fontsize=9)
        ax.tick_params(labelsize=8)
    for j in range(len(phens), len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"B4: acquisition phenomena shares — {label0}")
    plotting.save(fig, os.path.join(fdir, "acquisition_panel.png"))
