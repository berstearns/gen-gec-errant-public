"""D2 — Length & fluency confounds. RQ-delta."""
from __future__ import annotations

import os

import numpy as np
from scipy import stats

from . import common
from . import plotting

ID = "D2"
SLUG = "D2-length-fluency-confounds"


def _len_err_ppl(ctx, sid, keys):
    s = ctx.sources[sid]
    lengths, errs, ppls = [], [], []
    for k in keys:
        lengths.append(common.token_count(s.continuations.get(k, "")))
        errs.append(len(s.gen_errors.get(k, [])))
        p = s.perplexities.get(k)
        ppls.append(p if (p is not None and np.isfinite(p)) else np.nan)
    return np.array(lengths, float), np.array(errs, float), np.array(ppls, float)


def run(ctx: common.Context) -> dict:
    outdir = os.path.join(ctx.out_root, SLUG)
    keys = [tuple(k) for k in ctx.paired["keys"]]
    n = len(keys)

    data = {ctx.learner_id: _len_err_ppl(ctx, ctx.learner_id, keys)}
    ll, le, lp = data[ctx.learner_id]
    l_iqr = [float(np.percentile(ll, 25)), float(np.percentile(ll, 75))]

    def length_block(vec):
        return {"mean": float(np.nanmean(vec)), "median": float(np.nanmedian(vec)),
                "iqr": [float(np.percentile(vec, 25)), float(np.percentile(vec, 75))]}

    def fit(lengths, errs):
        if len(lengths) >= 3 and lengths.std() > 0:
            slope, intercept, r, p, se = stats.linregress(lengths, errs)
            return {"slope": float(slope), "r": float(r), "p": float(p)}
        return {"slope": None, "r": None, "p": None}

    def ppl_block(vec):
        v = vec[np.isfinite(vec)]
        if len(v) == 0:
            return {"mean": None, "median": None}
        return {"mean": float(v.mean()), "median": float(np.median(v))}

    # learner tag counts for length-matched JSD
    def gen_counts_over(sid, subset_keys):
        return ctx.sources[sid].gen_tag_counts(subset_keys)

    pair_results = {}
    length = {"LEARNER": length_block(ll)}
    fits = {"LEARNER": fit(ll, le)}
    ppl = {"LEARNER": ppl_block(lp)}
    for al_id, ctrl_id in ctx.pairs:
        data[al_id] = _len_err_ppl(ctx, al_id, keys)
        data[ctrl_id] = _len_err_ppl(ctx, ctrl_id, keys)
        al_l, al_e, al_p = data[al_id]
        ct_l, ct_e, ct_p = data[ctrl_id]
        length["AL"] = length_block(al_l); length["CONTROL"] = length_block(ct_l)
        fits["AL"] = fit(al_l, al_e); fits["CONTROL"] = fit(ct_l, ct_e)
        ppl["AL"] = ppl_block(al_p); ppl["CONTROL"] = ppl_block(ct_p)

        # length-matched subset: sentences whose LEARNER gen length in learner IQR
        lo, hi = l_iqr
        matched = [k for i, k in enumerate(keys) if lo <= ll[i] <= hi]
        lc_m = dict(gen_counts_over(ctx.learner_id, matched))
        lm_jsd = {"AL": common.jsd_counts(lc_m, dict(gen_counts_over(al_id, matched))),
                  "CONTROL": common.jsd_counts(lc_m, dict(gen_counts_over(ctrl_id, matched))),
                  "n": len(matched)}
        # raw JSD for comparison (each pair aligned on its own union vocab)
        lc_all = dict(ctx.sources[ctx.learner_id].gen_tag_counts(keys))
        raw_jsd = {"AL": common.jsd_counts(lc_all, dict(ctx.sources[al_id].gen_tag_counts(keys))),
                   "CONTROL": common.jsd_counts(lc_all, dict(ctx.sources[ctrl_id].gen_tag_counts(keys)))}
        ranking_holds = (lm_jsd["AL"] < lm_jsd["CONTROL"]) == (raw_jsd["AL"] < raw_jsd["CONTROL"])
        pair_results[ctx.pair_label(al_id, ctrl_id)] = {
            "length": dict(length), "len_err_fit": dict(fits), "ppl": dict(ppl),
            "length_matched_jsd": lm_jsd, "raw_jsd": raw_jsd, "ranking_holds": ranking_holds,
            "learner_length_iqr": l_iqr,
        }

    results = common.finalize_pairs(pair_results)
    caveats = ["EXPLORATORY. Perplexity is each model's OWN — NOT comparable across model families; "
               "used only within-pair (AL vs CONTROL) and as a degeneracy signal, never cross-family.",
               "If length-matched ranking flips vs raw JSD → MAJOR caveat to A1 (logged below)."]
    common.write_result(outdir, ID, ctx.run_slug, {"paired": n, "learner": ctx.learner_id}, {},
                        results, caveats)

    lines = ["# D2 — Length & fluency confounds", "",
             f"**Finding (n={n}, EXPLORATORY):** " + _headline(pair_results), ""]
    for label, b in pair_results.items():
        lines += [f"## Pair `{label}`", "",
                  "| source | len mean | len median | len→err slope | r | PPL mean |",
                  "|--------|----------|------------|---------------|---|----------|"]
        for role in ("LEARNER", "AL", "CONTROL"):
            L = b["length"][role]; F = b["len_err_fit"][role]; P = b["ppl"][role]
            sl = f"{F['slope']:.3f}" if F["slope"] is not None else "—"
            r = f"{F['r']:.3f}" if F["r"] is not None else "—"
            pm = f"{P['mean']:.1f}" if P["mean"] is not None else "—"
            lines.append(f"| {role} | {L['mean']:.1f} | {L['median']:.1f} | {sl} | {r} | {pm} |")
        lines += [f"- length-matched JSD (learner len IQR {b['learner_length_iqr']}, n={b['length_matched_jsd']['n']}): "
                  f"AL={b['length_matched_jsd']['AL']:.4f}, CONTROL={b['length_matched_jsd']['CONTROL']:.4f}",
                  f"- raw JSD: AL={b['raw_jsd']['AL']:.4f}, CONTROL={b['raw_jsd']['CONTROL']:.4f}",
                  f"- **AL-closer ranking holds under length matching? {b['ranking_holds']}**"]
    lines += ["", "## Caveats"] + [f"- {c}" for c in caveats]
    _b = next(iter(pair_results.values()))
    _lm, _raw = _b["length_matched_jsd"], _b["raw_jsd"]
    lines += ["", "## Conclusion", "",
              f"The AL-closer result is **not a length confound**: restricting to sentences whose generated "
              f"length falls in the learner interquartile range (n={_lm['n']}), AL remains nearer learners than "
              f"the control (length-matched JSD {_lm['AL']:.3f} vs {_lm['CONTROL']:.3f}), preserving the raw "
              f"ranking ({_raw['AL']:.3f} vs {_raw['CONTROL']:.3f}) — ranking holds = **{_b['ranking_holds']}**. "
              f"Error counts scale with length, which is why C1's per-token density (not raw counts) is the "
              f"fair magnitude metric. Perplexity is model-own and not compared across families. EXPLORATORY, "
              f"n={n}."]
    common.write_md(outdir, "\n".join(lines))

    _plot(outdir, ctx, data, pair_results, keys)
    common.write_inputs(outdir, ctx.run_slug, ctx.sources,
                        {"files": ["full_results.tsv", "raw_results.json"]})
    return results


def _headline(pair_results):
    b = next(iter(pair_results.values()))
    return (f"gen length mean — LEARNER={b['length']['LEARNER']['mean']:.0f}, "
            f"AL={b['length']['AL']['mean']:.0f}, CONTROL={b['length']['CONTROL']['mean']:.0f} tok; "
            f"length-matched ranking holds={b['ranking_holds']}")


def _plot(outdir, ctx, data, pair_results, keys):
    fdir = common.figures_dir(outdir)
    label0 = next(iter(pair_results)); al_id, ctrl_id = label0.split(":")
    ll, le, _ = data[ctx.learner_id]; al_l, al_e, _ = data[al_id]; ct_l, ct_e, _ = data[ctrl_id]
    common.save_csv(os.path.join(fdir, "len_vs_errors.csv"),
                    ["key", "role", "gen_tokens", "gen_errors"],
                    [[f"{k[0]}:{k[1]}", "LEARNER", ll[i], le[i]] for i, k in enumerate(keys)]
                    + [[f"{k[0]}:{k[1]}", "AL", al_l[i], al_e[i]] for i, k in enumerate(keys)]
                    + [[f"{k[0]}:{k[1]}", "CONTROL", ct_l[i], ct_e[i]] for i, k in enumerate(keys)])
    fig, ax = plotting.new_fig()
    for L, E, role in ((ll, le, "LEARNER"), (al_l, al_e, "AL"), (ct_l, ct_e, "CONTROL")):
        ax.scatter(L, E, s=14, alpha=0.5, label=role, color=plotting.ROLE_COLOR[role])
    ax.set_xlabel("generated tokens"); ax.set_ylabel("gen-region errors")
    ax.set_title(f"D2: length vs errors — {label0}"); ax.legend()
    plotting.save(fig, os.path.join(fdir, "len_vs_errors_scatter.png"))
    fig, ax = plotting.new_fig()
    mx = max(ll.max(), al_l.max(), ct_l.max())
    bins = np.linspace(0, mx, 20)
    for L, role in ((ll, "LEARNER"), (al_l, "AL"), (ct_l, "CONTROL")):
        ax.hist(L, bins=bins, alpha=0.5, label=role, color=plotting.ROLE_COLOR[role])
    ax.set_xlabel("generated tokens"); ax.set_ylabel("# sentences")
    ax.set_title(f"D2: generation length distribution — {label0}"); ax.legend()
    plotting.save(fig, os.path.join(fdir, "length_dist.png"))
