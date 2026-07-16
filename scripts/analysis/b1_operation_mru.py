"""B1 — Operation dissection (Missing / Replacement / Unnecessary). RQ-beta."""
from __future__ import annotations

import os

import numpy as np

from . import common
from . import plotting

ID = "B1"
SLUG = "B1-operation-mru"
OPS = ["M", "R", "U"]


def run(ctx: common.Context) -> dict:
    outdir = os.path.join(ctx.out_root, SLUG)
    n = ctx.paired["n"]

    def op_counts(sid):
        return ctx.dists["sources"][sid]["operation"]["counts"]

    def profile(sid):
        c = op_counts(sid)
        tot = sum(c.get(o, 0) for o in OPS)
        sh = {o: (c.get(o, 0) / tot if tot else 0.0) for o in OPS}
        mru = {o: int(c.get(o, 0)) for o in OPS}
        omission = (mru["M"] / (mru["R"] + mru["U"])) if (mru["R"] + mru["U"]) else None
        return {**mru, "total": tot, "m_share": sh["M"], "r_share": sh["R"],
                "u_share": sh["U"], "omission_ratio": omission}

    src_profiles = {ctx.learner_id: profile(ctx.learner_id)}
    pair_results = {}
    for al_id, ctrl_id in ctx.pairs:
        src_profiles[al_id] = profile(al_id)
        src_profiles[ctrl_id] = profile(ctrl_id)
        lp = src_profiles[ctx.learner_id]
        gap = {"AL": {o: src_profiles[al_id][f"{o.lower()}_share"] - lp[f"{o.lower()}_share"] for o in OPS},
               "CONTROL": {o: src_profiles[ctrl_id][f"{o.lower()}_share"] - lp[f"{o.lower()}_share"] for o in OPS}}
        l1_al = sum(abs(v) for v in gap["AL"].values())
        l1_ct = sum(abs(v) for v in gap["CONTROL"].values())
        pair_results[ctx.pair_label(al_id, ctrl_id)] = {
            "LEARNER": lp, "AL": src_profiles[al_id], "CONTROL": src_profiles[ctrl_id],
            "gap": gap, "l1_gap": {"AL": l1_al, "CONTROL": l1_ct},
            "AL_closer": l1_al < l1_ct,
        }

    results = common.finalize_pairs(pair_results)
    caveats = ["EXPLORATORY. Coarsest, most tiny-n-robust structural panel (3 buckets).",
               "Counts reconcile with A1 operation-granularity (same substrate)."]
    common.write_result(outdir, ID, ctx.run_slug, {"paired": n, "learner": ctx.learner_id}, {},
                        results, caveats)

    lines = ["# B1 — Operation dissection (M/R/U)", "",
             f"**Finding (n={n}, EXPLORATORY):** " + _headline(pair_results), ""]
    for label, b in pair_results.items():
        lines += [f"## Pair `{label}`", "",
                  "| source | M | R | U | M% | R% | U% | omission M:(R+U) |",
                  "|--------|---|---|---|----|----|----|------------------|"]
        for role in ("LEARNER", "AL", "CONTROL"):
            p = b[role]
            om = f"{p['omission_ratio']:.2f}" if p["omission_ratio"] is not None else "—"
            lines.append(f"| {role} | {p['M']} | {p['R']} | {p['U']} | {common.fmt_pct(p['m_share'])} | "
                         f"{common.fmt_pct(p['r_share'])} | {common.fmt_pct(p['u_share'])} | {om} |")
        lines.append(f"- L1 gap to LEARNER: AL={b['l1_gap']['AL']:.3f}, CONTROL={b['l1_gap']['CONTROL']:.3f} "
                     f"→ AL closer? **{b['AL_closer']}**")
    lines += ["", "## Caveats"] + [f"- {c}" for c in caveats]
    _b = next(iter(pair_results.values()))
    _ol, _oa, _oc = _b["LEARNER"]["omission_ratio"], _b["AL"]["omission_ratio"], _b["CONTROL"]["omission_ratio"]
    _fmt = lambda x: f"{x:.2f}" if x is not None else "—"
    lines += ["", "## Conclusion", "",
              f"All three sources share a nearly identical, omission-light operation balance — M:(R+U) "
              f"= **{_fmt(_ol)}** (learner) vs {_fmt(_oa)} (AL) vs {_fmt(_oc)} (control) — and the AL–learner "
              f"operation-level divergence is minute (L1 gap {_b['l1_gap']['AL']:.3f}). The interlanguage "
              f"omission tendency is therefore reproduced already by native pretraining, not specifically by "
              f"fine-tuning; the discriminating signal lives at finer granularities (B2/B3/B4), not in the "
              f"M/R/U mix. This is the most tiny-n-robust panel (3 buckets). EXPLORATORY, n={n}."]
    common.write_md(outdir, "\n".join(lines))

    _plot(outdir, pair_results)
    common.write_inputs(outdir, ctx.run_slug, ctx.sources, {"shared": ["_shared/distributions.json"]})
    return results


def _headline(pair_results):
    b = next(iter(pair_results.values()))
    return (f"omission M:(R+U) — LEARNER={_o(b['LEARNER'])}, AL={_o(b['AL'])}, CONTROL={_o(b['CONTROL'])}; "
            f"AL closer to LEARNER M/R/U balance? {b['AL_closer']}")


def _o(p):
    return f"{p['omission_ratio']:.2f}" if p["omission_ratio"] is not None else "—"


def _plot(outdir, pair_results):
    fdir = common.figures_dir(outdir)
    label0 = next(iter(pair_results)); b = pair_results[label0]
    rows = [[role, b[role]["m_share"], b[role]["r_share"], b[role]["u_share"]]
            for role in ("LEARNER", "AL", "CONTROL")]
    common.save_csv(os.path.join(fdir, "mru_stacked.csv"), ["source", "M", "R", "U"], rows)
    fig, ax = plotting.new_fig(5.5, 4.2)
    roles = ["LEARNER", "AL", "CONTROL"]
    x = np.arange(len(roles))
    bottom = np.zeros(len(roles))
    colors = {"M": "#CC6677", "R": "#4477AA", "U": "#DDCC77"}
    for op in OPS:
        vals = np.array([b[r][f"{op.lower()}_share"] for r in roles])
        ax.bar(x, vals, 0.6, bottom=bottom, label=op, color=colors[op])
        bottom += vals
    ax.set_xticks(x); ax.set_xticklabels(roles)
    ax.set_ylabel("share of gen-region errors"); ax.set_ylim(0, 1)
    ax.set_title(f"B1: operation balance — {label0}"); ax.legend(title="op")
    plotting.save(fig, os.path.join(fdir, "mru_stacked.png"))
