"""C1 — Error magnitude & density. RQ-gamma (rate, not shape)."""
from __future__ import annotations

import os

import numpy as np
from scipy import stats

from . import common
from . import plotting


ID = "C1"
SLUG = "C1-magnitude-density"


def _per_sentence(ctx, sid, keys):
    """(err_counts, densities) arrays aligned to keys; density = gen errors / gen tokens."""
    s = ctx.sources[sid]
    counts, dens = [], []
    for k in keys:
        e = len(s.gen_errors.get(k, []))
        tok = common.token_count(s.continuations.get(k, ""))
        counts.append(e)
        dens.append(e / tok if tok > 0 else 0.0)
    return np.array(counts, float), np.array(dens, float)


def run(ctx: common.Context) -> dict:
    outdir = os.path.join(ctx.out_root, SLUG)
    keys = [tuple(k) for k in ctx.paired["keys"]]
    n = len(keys)

    per = {ctx.learner_id: _per_sentence(ctx, ctx.learner_id, keys)}
    src_stats = {}
    def stat_block(sid):
        c, d = per[sid]
        return {"err_per_sent": {"mean": float(c.mean()), "median": float(np.median(c)),
                                 "std": float(c.std(ddof=0))},
                "error_rate": float((c >= 1).mean()),
                "density": {"mean": float(d.mean()), "median": float(np.median(d))}}
    src_stats["LEARNER"] = stat_block(ctx.learner_id)

    lc, _ = per[ctx.learner_id]
    pair_results = {}
    for al_id, ctrl_id in ctx.pairs:
        per[al_id] = _per_sentence(ctx, al_id, keys)
        per[ctrl_id] = _per_sentence(ctx, ctrl_id, keys)
        ac, _ = per[al_id]; cc, _ = per[ctrl_id]
        ks_al = stats.ks_2samp(lc, ac)
        ks_ct = stats.ks_2samp(lc, cc)
        pair_results[ctx.pair_label(al_id, ctrl_id)] = {
            "LEARNER": src_stats["LEARNER"], "AL": stat_block(al_id), "CONTROL": stat_block(ctrl_id),
            "ks": {"AL": {"stat": float(ks_al.statistic), "p": float(ks_al.pvalue)},
                   "CONTROL": {"stat": float(ks_ct.statistic), "p": float(ks_ct.pvalue)}},
        }

    results = common.finalize_pairs(pair_results)
    caveats = ["EXPLORATORY. density denom = spaCy(en_core_web_sm) token count of the generated "
               "continuation (matches ERRANT tokenization).",
               "KS p-values unreliable at tiny n → G1. Means reconcile with <model>_summary.json."]
    common.write_result(outdir, ID, ctx.run_slug, {"paired": n, "learner": ctx.learner_id},
                        {"density_tokenizer": "en_core_web_sm"}, results, caveats)

    lines = ["# C1 — Error magnitude & density", "",
             f"**Finding (n={n}, EXPLORATORY):** " + _headline(pair_results), ""]
    for label, b in pair_results.items():
        lines += [f"## Pair `{label}`", "",
                  "| source | errs/sent (mean) | median | std | error rate | density (mean) |",
                  "|--------|------------------|--------|-----|------------|----------------|"]
        for role in ("LEARNER", "AL", "CONTROL"):
            p = b[role]
            lines.append(f"| {role} | {p['err_per_sent']['mean']:.3f} | {p['err_per_sent']['median']:.1f} | "
                         f"{p['err_per_sent']['std']:.3f} | {common.fmt_pct(p['error_rate'])} | "
                         f"{p['density']['mean']:.4f} |")
        lines.append(f"- KS(err-count) vs LEARNER: AL stat={b['ks']['AL']['stat']:.3f} (p={b['ks']['AL']['p']:.3f}), "
                     f"CONTROL stat={b['ks']['CONTROL']['stat']:.3f} (p={b['ks']['CONTROL']['p']:.3f})")
    lines += ["", "## Caveats"] + [f"- {c}" for c in caveats]
    _b = next(iter(pair_results.values()))
    _lr, _ar, _cr = (_b[r]["err_per_sent"]["mean"] for r in ("LEARNER", "AL", "CONTROL"))
    _ld, _ad, _cd = (_b[r]["density"]["mean"] for r in ("LEARNER", "AL", "CONTROL"))
    _rate = _ar / _lr if _lr else float("inf")
    _dens = _ad / _ld if _ld else float("inf")
    _dens_ct = _cd / _ld if _ld else float("inf")
    lines += ["", "## Conclusion", "",
              f"Shape and magnitude are decoupled. AL errs **{_rate:.2f}× as often per sentence** as authentic "
              f"learners ({_ar:.2f} vs {_lr:.2f}) yet at only **{_dens:.2f}× their per-token density** "
              f"({_ad:.3f} vs {_ld:.3f}) — a longer, less error-dense output, i.e. a *diluted* learner. On the "
              f"length-controlled density metric AL ({_dens:.2f}×) tracks learners more tightly than the "
              f"control does ({_dens_ct:.2f}×). Rate and density must be reported separately: learner-like "
              f"shape (A1) does not imply learner-like intensity. KS p unreliable at n={n} → G1. EXPLORATORY."]
    common.write_md(outdir, "\n".join(lines))

    _plot(outdir, ctx, per, pair_results, keys)
    common.write_inputs(outdir, ctx.run_slug, ctx.sources,
                        {"files": ["errors_long_format.tsv", "raw_results.json (continuations)"]})
    return results


def _headline(pair_results):
    b = next(iter(pair_results.values()))
    return (f"errs/sent — LEARNER={b['LEARNER']['err_per_sent']['mean']:.2f}, "
            f"AL={b['AL']['err_per_sent']['mean']:.2f}, CONTROL={b['CONTROL']['err_per_sent']['mean']:.2f}; "
            f"density LEARNER={b['LEARNER']['density']['mean']:.3f}, AL={b['AL']['density']['mean']:.3f}, "
            f"CONTROL={b['CONTROL']['density']['mean']:.3f}")


def _plot(outdir, ctx, per, pair_results, keys):
    fdir = common.figures_dir(outdir)
    label0 = next(iter(pair_results)); al_id, ctrl_id = label0.split(":")
    lc, ld = per[ctx.learner_id]; ac, ad = per[al_id]; cc, cd = per[ctrl_id]
    common.save_csv(os.path.join(fdir, "errcount_hist.csv"), ["key", "LEARNER", "AL", "CONTROL"],
                    [[f"{k[0]}:{k[1]}", int(lc[i]), int(ac[i]), int(cc[i])] for i, k in enumerate(keys)])
    # overlaid error-count hist
    fig, ax = plotting.new_fig()
    mx = int(max(lc.max(), ac.max(), cc.max()))
    bins = np.arange(0, mx + 2) - 0.5
    for vec, role in ((lc, "LEARNER"), (ac, "AL"), (cc, "CONTROL")):
        ax.hist(vec, bins=bins, alpha=0.5, label=role, color=plotting.ROLE_COLOR[role])
    ax.set_xlabel("gen-region errors per sentence"); ax.set_ylabel("# sentences")
    ax.set_title(f"C1: per-sentence error counts — {label0}"); ax.legend()
    plotting.save(fig, os.path.join(fdir, "errcount_hist.png"))
    # density box
    fig, ax = plotting.new_fig(5.5, 4.2)
    bp = ax.boxplot([ld, ad, cd], labels=["LEARNER", "AL", "CONTROL"], patch_artist=True)
    for patch, role in zip(bp["boxes"], ("LEARNER", "AL", "CONTROL")):
        patch.set_facecolor(plotting.ROLE_COLOR[role]); patch.set_alpha(0.6)
    ax.set_ylabel("error density (errors / gen tokens)")
    ax.set_title(f"C1: error density — {label0}")
    plotting.save(fig, os.path.join(fdir, "density_box.png"))
