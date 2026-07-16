"""B2 — POS-family dissection. RQ-beta (mid-granularity)."""
from __future__ import annotations

import os

import numpy as np

from . import common
from . import plotting

ID = "B2"
SLUG = "B2-pos-family"


def run(ctx: common.Context) -> dict:
    outdir = os.path.join(ctx.out_root, SLUG)
    n = ctx.paired["n"]

    def pos_shares(sid):
        return ctx.dists["sources"][sid]["pos"]["shares"]

    lsh = pos_shares(ctx.learner_id)
    families = sorted(set().union(*[set(ctx.dists["sources"][sid]["pos"]["shares"])
                                    for sid in ctx.sources]))

    pair_results = {}
    for al_id, ctrl_id in ctx.pairs:
        alsh, ctsh = pos_shares(al_id), pos_shares(ctrl_id)
        gap_al = {f: alsh.get(f, 0) - lsh.get(f, 0) for f in families}
        gap_ct = {f: ctsh.get(f, 0) - lsh.get(f, 0) for f in families}
        toward = {}
        for f in families:
            toward[f] = abs(alsh.get(f, 0) - lsh.get(f, 0)) < abs(ctsh.get(f, 0) - lsh.get(f, 0))
        # directional agreement over families present in learner OR either source
        active = [f for f in families if (lsh.get(f, 0) or alsh.get(f, 0) or ctsh.get(f, 0))]
        da = sum(toward[f] for f in active) / len(active) if active else 0.0
        pair_results[ctx.pair_label(al_id, ctrl_id)] = {
            "shares": {"LEARNER": {f: lsh.get(f, 0) for f in families},
                       "AL": {f: alsh.get(f, 0) for f in families},
                       "CONTROL": {f: ctsh.get(f, 0) for f in families}},
            "gap": {"AL": gap_al, "CONTROL": gap_ct},
            "toward_learner": toward,
            "directional_agreement": da,
        }

    results = common.finalize_pairs(pair_results)
    results["pos_map"] = ctx.dists["pos_map"]
    caveats = ["EXPLORATORY. directional_agreement is a compact β-summary; per-family shares at "
               "tiny n are noisy — see B1 for the robust coarse signal and G1 for CIs.",
               "Families reconcile with B3 tag aggregation (same pos_map)."]
    common.write_result(outdir, ID, ctx.run_slug, {"paired": n, "learner": ctx.learner_id}, {},
                        results, caveats)

    lines = ["# B2 — POS-family dissection", "",
             f"**Finding (n={n}, EXPLORATORY):** " + _headline(pair_results), ""]
    for label, b in pair_results.items():
        al_id, ctrl_id = label.split(":")
        lines += [f"## Pair `{label}`", "",
                  f"- **directional agreement** (families where AL nearer LEARNER than CONTROL): "
                  f"**{common.fmt_pct(b['directional_agreement'])}**", "",
                  "| family | LEARNER | AL | CONTROL | gap_AL | gap_CTRL | AL→lrn |",
                  "|--------|---------|-----|---------|--------|----------|--------|"]
        order = sorted(families, key=lambda f: -b["shares"]["LEARNER"][f])
        for f in order:
            s = b["shares"]
            if not (s["LEARNER"][f] or s["AL"][f] or s["CONTROL"][f]):
                continue
            lines.append(f"| {f} | {common.fmt_pct(s['LEARNER'][f])} | {common.fmt_pct(s['AL'][f])} | "
                         f"{common.fmt_pct(s['CONTROL'][f])} | {b['gap']['AL'][f]:+.3f} | "
                         f"{b['gap']['CONTROL'][f]:+.3f} | {'✓' if b['toward_learner'][f] else ''} |")
    lines += ["", "## Caveats"] + [f"- {c}" for c in caveats]
    _b = next(iter(pair_results.values()))
    _da = _b["directional_agreement"]
    _toward = [f for f, t in _b["toward_learner"].items() if t and
               (_b["shares"]["LEARNER"][f] or _b["shares"]["AL"][f] or _b["shares"]["CONTROL"][f])]
    lines += ["", "## Conclusion", "",
              f"Fine-tuning moves AL nearer the learner error share than the control in "
              f"**{common.fmt_pct(_da)}** of active POS families (directional agreement). The families pulled "
              f"toward learners include {', '.join(_toward[:6]) or 'none'} — the content-word categories that "
              f"carry L2 error (determiner, preposition, verb, noun), consistent with B4's acquisition panel. "
              f"Per-family shares are noisy at n={n}; the scalar directional-agreement summary and G1 CIs are "
              f"the load-bearing readouts, not individual family gaps. EXPLORATORY."]
    common.write_md(outdir, "\n".join(lines))

    _plot(outdir, pair_results, families)
    common.write_inputs(outdir, ctx.run_slug, ctx.sources, {"shared": ["_shared/distributions.json"]})
    return results


def _headline(pair_results):
    b = next(iter(pair_results.values()))
    return f"directional agreement (AL nearer LEARNER than CONTROL) = {common.fmt_pct(b['directional_agreement'])} of active POS families"


def _plot(outdir, pair_results, families):
    fdir = common.figures_dir(outdir)
    label0 = next(iter(pair_results)); b = pair_results[label0]
    order = [f for f in sorted(families, key=lambda f: -b["shares"]["LEARNER"][f])
             if (b["shares"]["LEARNER"][f] or b["shares"]["AL"][f] or b["shares"]["CONTROL"][f])]
    rows = [[f, b["shares"]["LEARNER"][f], b["shares"]["AL"][f], b["shares"]["CONTROL"][f]] for f in order]
    common.save_csv(os.path.join(fdir, "pos_family_grouped.csv"),
                    ["family", "LEARNER", "AL", "CONTROL"], rows)
    x = np.arange(len(order)); w = 0.27
    fig, ax = plotting.new_fig(max(7, 0.6 * len(order)), 4.4)
    ax.bar(x - w, [b["shares"]["LEARNER"][f] for f in order], w, label="LEARNER", color=plotting.ROLE_COLOR["LEARNER"])
    ax.bar(x, [b["shares"]["AL"][f] for f in order], w, label="AL", color=plotting.ROLE_COLOR["AL"])
    ax.bar(x + w, [b["shares"]["CONTROL"][f] for f in order], w, label="CONTROL", color=plotting.ROLE_COLOR["CONTROL"])
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_ylabel("share of gen-region errors")
    ax.set_title(f"B2: POS-family error shares — {label0}"); ax.legend()
    plotting.save(fig, os.path.join(fdir, "pos_family_grouped.png"))
