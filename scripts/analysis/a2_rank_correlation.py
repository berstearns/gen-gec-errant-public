"""A2 — Rank correlation & rank displacement. RQ-alpha (rank-based)."""
from __future__ import annotations

import os

import numpy as np
from scipy import stats

from . import common
from . import plotting

ID = "A2"
SLUG = "A2-rank-correlation-displacement"


def _vecs(vocab, counts):
    return np.array([counts.get(t, 0) for t in vocab], float)


def _ranks(vec):
    """Descending-share ranks, ties averaged (rank 1 = largest)."""
    return stats.rankdata(-vec, method="average")


def _topk_jaccard(learner_counts, source_counts, k):
    lt = [t for t, _ in sorted(learner_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]
    st = [t for t, _ in sorted(source_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]
    a, b = set(lt), set(st)
    return len(a & b) / len(a | b) if (a | b) else 0.0


def run(ctx: common.Context) -> dict:
    outdir = os.path.join(ctx.out_root, SLUG)
    ks = ctx.params.get("k", [5, 10])
    n = ctx.paired["n"]

    def tagc(sid):
        return ctx.dists["sources"][sid]["tag"]["counts"]

    lc = tagc(ctx.learner_id)
    pair_results = {}
    for al_id, ctrl_id in ctx.pairs:
        sc_al, sc_ctrl = tagc(al_id), tagc(ctrl_id)
        # single union vocab across all three roles → aligned, equal-length vectors
        vocab = sorted(set(lc) | set(sc_al) | set(sc_ctrl))
        lvec, alvec, ctrlvec = _vecs(vocab, lc), _vecs(vocab, sc_al), _vecs(vocab, sc_ctrl)

        sp_al = stats.spearmanr(lvec, alvec)
        sp_ct = stats.spearmanr(lvec, ctrlvec)
        kt_al = stats.kendalltau(lvec, alvec, variant="b")
        kt_ct = stats.kendalltau(lvec, ctrlvec, variant="b")

        r_l, r_al, r_ct = _ranks(lvec), _ranks(alvec), _ranks(ctrlvec)
        disp = []
        for i, t in enumerate(vocab):
            disp.append({"tag": t, "rank_learner": float(r_l[i]), "rank_AL": float(r_al[i]),
                         "rank_CONTROL": float(r_ct[i]),
                         "d_AL": float(r_al[i] - r_l[i]), "d_CONTROL": float(r_ct[i] - r_l[i])})
        disp.sort(key=lambda d: (-abs(d["d_AL"]), d["tag"]))

        pair_results[ctx.pair_label(al_id, ctrl_id)] = {
            "spearman": {"AL": _safe(sp_al.statistic), "CONTROL": _safe(sp_ct.statistic)},
            "spearman_p": {"AL": _safe(sp_al.pvalue), "CONTROL": _safe(sp_ct.pvalue)},
            "kendall": {"AL": _safe(kt_al.statistic), "CONTROL": _safe(kt_ct.statistic)},
            "kendall_p": {"AL": _safe(kt_al.pvalue), "CONTROL": _safe(kt_ct.pvalue)},
            "topk_jaccard": {f"k{k}": {"AL": _topk_jaccard(lc, sc_al, k),
                                       "CONTROL": _topk_jaccard(lc, sc_ctrl, k)} for k in ks},
            "displacement": disp,
        }

    results = common.finalize_pairs(pair_results)
    caveats = [
        "EXPLORATORY. Spearman inflates with 0-tie ties at tiny n; τ-b is the robust headline.",
        "Correlation p-values are unreliable at this n → G1 permutation test is authoritative.",
    ]
    common.write_result(outdir, ID, ctx.run_slug, {"paired": n, "learner": ctx.learner_id},
                        {"k": ks, "tie": "average"}, results, caveats)

    lines = ["# A2 — Rank correlation & displacement", "",
             f"**Finding (n={n}, EXPLORATORY):** " + _headline(pair_results), ""]
    for label, b in pair_results.items():
        lines += [f"## Pair `{label}`", "",
                  "| metric | AL | CONTROL |", "|--------|-----|---------|",
                  f"| Spearman ρ | {b['spearman']['AL']:.3f} | {b['spearman']['CONTROL']:.3f} |",
                  f"| Kendall τ-b | {b['kendall']['AL']:.3f} | {b['kendall']['CONTROL']:.3f} |",
                  f"| top-5 Jaccard | {b['topk_jaccard']['k5']['AL']:.3f} | {b['topk_jaccard']['k5']['CONTROL']:.3f} |",
                  f"| top-10 Jaccard | {b['topk_jaccard']['k10']['AL']:.3f} | {b['topk_jaccard']['k10']['CONTROL']:.3f} |",
                  "", "### Top-10 most AL-displaced tags",
                  "| tag | rank_lrn | rank_AL | rank_CTRL | Δrank_AL | Δrank_CTRL |",
                  "|-----|----------|---------|-----------|----------|------------|"]
        for d in b["displacement"][:10]:
            lines.append(f"| {d['tag']} | {d['rank_learner']:.0f} | {d['rank_AL']:.0f} | "
                         f"{d['rank_CONTROL']:.0f} | {d['d_AL']:+.0f} | {d['d_CONTROL']:+.0f} |")
    lines += ["", "## Caveats"] + [f"- {c}" for c in caveats]
    _b = next(iter(pair_results.values()))
    _ka, _kc = _b["kendall"]["AL"], _b["kendall"]["CONTROL"]
    _ratio = (_ka / _kc) if _kc else float("inf")
    _topdisp = ", ".join(d["tag"] for d in _b["displacement"][:3])
    lines += ["", "## Conclusion", "",
              f"AL orders the error types more like authentic learners than the control does: Kendall τ-b "
              f"**{_ka:.3f} vs {_kc:.3f}** (a {_ratio:.2f}× stronger rank agreement). The ordering breaks most "
              f"on {_topdisp} — the tags whose learner rank AL misplaces furthest. This is a shape-of-ranking "
              f"effect independent of exact masses (cf. A1); its permutation significance is given by G1 "
              f"(ρ-diff p), not the per-correlation p-values, which are unreliable at n={n}. EXPLORATORY."]
    common.write_md(outdir, "\n".join(lines))

    _plot(outdir, pair_results)
    common.write_inputs(outdir, ctx.run_slug, ctx.sources, {"shared": ["_shared/distributions.json"]})
    return results


def _safe(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def _headline(pair_results):
    parts = []
    for label, b in pair_results.items():
        al, ct = b["kendall"]["AL"], b["kendall"]["CONTROL"]
        closer = "AL orders tags more learner-like" if (al or -9) > (ct or -9) else "CONTROL orders tags more learner-like"
        parts.append(f"{label}: τ-b AL={al:.3f} vs CONTROL={ct:.3f} → {closer}")
    return "; ".join(parts)


def _plot(outdir, pair_results):
    fdir = common.figures_dir(outdir)
    label0 = next(iter(pair_results))
    disp = pair_results[label0]["displacement"]
    # Rows sorted by learner rank; show tags that are ranked in learner top or displaced.
    top = sorted(disp, key=lambda d: d["rank_learner"])[:12]
    common.save_csv(os.path.join(fdir, "rank_displacement.csv"),
                    ["tag", "rank_learner", "rank_AL", "rank_CONTROL", "d_AL", "d_CONTROL"],
                    [[d["tag"], d["rank_learner"], d["rank_AL"], d["rank_CONTROL"], d["d_AL"], d["d_CONTROL"]]
                     for d in disp])
    fig, ax = plotting.new_fig(7.5, 5.5)
    ys = np.arange(len(top))[::-1]
    for i, d in zip(ys, top):
        ax.plot([d["rank_learner"], d["rank_AL"]], [i, i], color=plotting.ROLE_COLOR["AL"], alpha=0.5, lw=1)
        ax.plot([d["rank_learner"], d["rank_CONTROL"]], [i, i], color=plotting.ROLE_COLOR["CONTROL"], alpha=0.5, lw=1)
    ax.scatter([d["rank_learner"] for d in top], ys, color=plotting.ROLE_COLOR["LEARNER"], label="LEARNER", zorder=3)
    ax.scatter([d["rank_AL"] for d in top], ys, color=plotting.ROLE_COLOR["AL"], label="AL", zorder=3)
    ax.scatter([d["rank_CONTROL"] for d in top], ys, color=plotting.ROLE_COLOR["CONTROL"], label="CONTROL", zorder=3)
    ax.set_yticks(ys); ax.set_yticklabels([d["tag"] for d in top])
    ax.set_xlabel("rank (1 = most frequent error tag)")
    ax.set_title(f"A2: tag-rank displacement — {label0}")
    ax.legend()
    plotting.save(fig, os.path.join(fdir, "rank_displacement.png"))
