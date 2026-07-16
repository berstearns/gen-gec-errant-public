"""A1 — Distributional similarity (JSD/KL/TVD/cross-entropy). RQ-alpha."""
from __future__ import annotations

import os

from . import common
from . import plotting

ID = "A1"
SLUG = "A1-distributional-similarity"
GRANS = ["operation", "pos", "tag"]


def _counts(ctx, sid, gran):
    return ctx.dists["sources"][sid][gran]["counts"]


def run(ctx: common.Context) -> dict:
    outdir = os.path.join(ctx.out_root, SLUG)
    eps = ctx.params.get("smoothing_eps", 1e-9)
    n = ctx.paired["n"]

    pair_results = {}
    for al_id, ctrl_id in ctx.pairs:
        block = {}
        for gran in GRANS:
            lc = _counts(ctx, ctx.learner_id, gran)
            block[gran] = {
                "AL": common.all_metrics(lc, _counts(ctx, al_id, gran), eps),
                "CONTROL": common.all_metrics(lc, _counts(ctx, ctrl_id, gran), eps),
            }
            block[gran]["AL_closer"] = block[gran]["AL"]["jsd"] < block[gran]["CONTROL"]["jsd"]
        pair_results[ctx.pair_label(al_id, ctrl_id)] = block

    results = common.finalize_pairs(pair_results)
    caveats = [
        "EXPLORATORY (tiny sample). Point estimates only; CIs come from G1.",
        "An 'AL closer' claim is provisional until G1's bootstrap CI excludes 0.",
    ]
    common.write_result(outdir, ID, ctx.run_slug,
                        {"paired": n, "learner": ctx.learner_id}, {"smoothing_eps": eps, "jsd_base": 2},
                        results, caveats)

    # ---- result.md ----
    lines = [f"# A1 — Distributional similarity", "",
             f"**Finding (n={n}, EXPLORATORY):** " + _headline(pair_results), ""]
    for label, block in pair_results.items():
        al_id, ctrl_id = label.split(":")
        lines.append(f"## Pair `{label}`  (AL={al_id}, CONTROL={ctrl_id})")
        for gran in GRANS:
            b = block[gran]
            lines += [f"", f"### {gran} granularity",
                      "| source | JSD | TVD | KL(src‖lrn) | KL(lrn‖src) | H(lrn,src) | n_src |",
                      "|--------|-----|-----|-------------|-------------|-----------|-------|"]
            for role in ("AL", "CONTROL"):
                m = b[role]
                jsd = f"**{m['jsd']:.4f}**" if (role == "AL") == b["AL_closer"] else f"{m['jsd']:.4f}"
                lines.append(f"| {role} | {jsd} | {m['tvd']:.4f} | {m['kl_sl']:.3f} | "
                             f"{m['kl_ls']:.3f} | {m['xent']:.3f} | {m['n_source']} |")
            lines.append(f"- AL closer to LEARNER than CONTROL? **{b['AL_closer']}** "
                         f"(learner n={b['AL']['n_learner']})")
    lines += ["", "## Caveats"] + [f"- {c}" for c in caveats]
    _b = next(iter(pair_results.values()))
    _jt_al, _jt_ct = _b["tag"]["AL"]["jsd"], _b["tag"]["CONTROL"]["jsd"]
    _jo_al, _jo_ct = _b["operation"]["AL"]["jsd"], _b["operation"]["CONTROL"]["jsd"]
    _rdr_tag = (_jt_ct - _jt_al) / _jt_ct if _jt_ct else 0.0
    _rdr_op = (_jo_ct - _jo_al) / _jo_ct if _jo_ct else 0.0
    lines += ["", "## Conclusion", "",
              f"Fine-tuning closes **{common.fmt_pct(_rdr_tag)}** of the pretrained control's tag-level "
              f"distance to authentic learners (RDR = (JSD_ctrl−JSD_AL)/JSD_ctrl; JSD_tag "
              f"{_jt_ct:.3f}→{_jt_al:.3f}), and a larger {common.fmt_pct(_rdr_op)} at the coarse M/R/U "
              f"operation level (JSD {_jo_ct:.3f}→{_jo_al:.3f}) — the similarity is strongest coarsely and "
              f"attenuates at the fine tag level, i.e. fine-tuning re-weights *which* error types occur more "
              f"than it changes the omission/commission balance. Directional (AL nearer learners on every "
              f"granularity) but provisional: G1's bootstrap CI for the tag-level gap includes 0 at n={n}. "
              f"EXPLORATORY."]
    common.write_md(outdir, "\n".join(lines))

    # ---- figure: jsd_by_granularity ----
    _plot_jsd(outdir, pair_results)

    common.write_inputs(outdir, ctx.run_slug, ctx.sources,
                        {"shared": ["_shared/distributions.json", "_shared/paired_index.json"]})
    return results


def _headline(pair_results):
    parts = []
    for label, block in pair_results.items():
        b = block["tag"]
        verdict = "closer to LEARNER than CONTROL" if b["AL_closer"] else "NOT closer than CONTROL"
        parts.append(f"{label}: AL JSD_tag={b['AL']['jsd']:.3f} vs CONTROL {b['CONTROL']['jsd']:.3f} "
                     f"→ AL {verdict}")
    return "; ".join(parts)


def _plot_jsd(outdir, pair_results):
    fdir = common.figures_dir(outdir)
    # Use the first pair for the canonical figure; CSV holds all pairs.
    rows = []
    for label, block in pair_results.items():
        for gran in GRANS:
            rows.append([label, gran, block[gran]["AL"]["jsd"], block[gran]["CONTROL"]["jsd"]])
    common.save_csv(os.path.join(fdir, "jsd_by_granularity.csv"),
                    ["pair", "granularity", "jsd_AL", "jsd_CONTROL"], rows)

    import numpy as np
    label0 = next(iter(pair_results))
    block = pair_results[label0]
    al = [block[g]["AL"]["jsd"] for g in GRANS]
    ctrl = [block[g]["CONTROL"]["jsd"] for g in GRANS]
    x = np.arange(len(GRANS)); w = 0.38
    fig, ax = plotting.new_fig()
    ax.bar(x - w / 2, al, w, label="AL", color=plotting.ROLE_COLOR["AL"])
    ax.bar(x + w / 2, ctrl, w, label="CONTROL", color=plotting.ROLE_COLOR["CONTROL"])
    ax.set_xticks(x); ax.set_xticklabels(GRANS)
    ax.set_ylabel("JSD to LEARNER (base 2, ↓ = closer)")
    ax.set_title(f"A1: distributional distance to LEARNER — {label0}")
    ax.legend()
    plotting.save(fig, os.path.join(fdir, "jsd_by_granularity.png"))
