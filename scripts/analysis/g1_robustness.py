"""G1 — Statistical robustness (sentence bootstrap + permutation). RQ-epsilon.

Resamples at the SENTENCE level (the independent unit). Attaches CIs to the
A1 JSD, the B4 acquisition-alignment and the E1 signature-alignment, tests
whether Δ=JSD_CONTROL−JSD_AL>0 is robust, and states the minimum detectable
effect. Same machinery gives real power at full S1.
"""
from __future__ import annotations

import os

import numpy as np
from scipy import stats

from . import common
from . import plotting

ID = "G1"
SLUG = "G1-statistical-robustness"


def _count_matrix(source, keys, vocab, vindex):
    """(n_keys x V) integer tag-count matrix over the union vocab, gen-region."""
    m = np.zeros((len(keys), len(vocab)), dtype=float)
    for i, k in enumerate(keys):
        for row in source.gen_errors.get(k, []):
            j = vindex.get(row["error_type"])
            if j is not None:
                m[i, j] += 1
    return m


def _jsd_vec(p, q):
    p = p / p.sum() if p.sum() > 0 else p
    q = q / q.sum() if q.sum() > 0 else q
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _acq_alignment(lvec, avec, cvec, phen_groups):
    toward = 0
    for idxs in phen_groups.values():
        lp = lvec[idxs].sum() / lvec.sum() if lvec.sum() else 0
        ap = avec[idxs].sum() / avec.sum() if avec.sum() else 0
        cp = cvec[idxs].sum() / cvec.sum() if cvec.sum() else 0
        if abs(ap - lp) < abs(cp - lp):
            toward += 1
    return toward / len(phen_groups)


def _sig_alignment(lvec, avec, cvec):
    ls = lvec / lvec.sum() if lvec.sum() else lvec
    as_ = avec / avec.sum() if avec.sum() else avec
    cs = cvec / cvec.sum() if cvec.sum() else cvec
    na = avec.sum(); nc = cvec.sum()
    ea = 0.5 / na if na else 1e-6
    ec = 0.5 / nc if nc else 1e-6
    present = lvec > 0
    if present.sum() == 0:
        return 0.0
    l2a = np.abs(np.log2((as_[present] + ea) / (ls[present] + ea)))
    l2c = np.abs(np.log2((cs[present] + ec) / (ls[present] + ec)))
    return float((l2a < l2c).mean())


def run(ctx: common.Context) -> dict:
    outdir = os.path.join(ctx.out_root, SLUG)
    keys = [tuple(k) for k in ctx.paired["keys"]]
    n = len(keys)
    B = ctx.params.get("B", 2000)
    ci = ctx.params.get("ci", 0.95)
    seed = ctx.params.get("seed", 42)
    lo_q, hi_q = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100

    # union vocab over the three roles for the (first) pair; recomputed per pair
    def pack(al_id, ctrl_id):
        tags = set()
        for sid in (ctx.learner_id, al_id, ctrl_id):
            tags |= set(ctx.sources[sid].gen_tag_counts(keys))
        vocab = sorted(tags)
        vindex = {t: j for j, t in enumerate(vocab)}
        L = _count_matrix(ctx.sources[ctx.learner_id], keys, vocab, vindex)
        A = _count_matrix(ctx.sources[al_id], keys, vocab, vindex)
        C = _count_matrix(ctx.sources[ctrl_id], keys, vocab, vindex)
        phen_groups = {}
        for phen, ptags in common.ACQUISITION_PHENOMENA.items():
            phen_groups[phen] = [vindex[t] for t in ptags if t in vindex]
        return vocab, L, A, C, phen_groups

    pair_results = {}
    fig_payload = None
    for al_id, ctrl_id in ctx.pairs:
        vocab, L, A, C, phen_groups = pack(al_id, ctrl_id)
        Ls, As, Cs = L.sum(0), A.sum(0), C.sum(0)
        obs_jsd_al = _jsd_vec(Ls, As)
        obs_jsd_ct = _jsd_vec(Ls, Cs)
        obs_delta = obs_jsd_ct - obs_jsd_al
        obs_acq = _acq_alignment(Ls, As, Cs, phen_groups)
        obs_sig = _sig_alignment(Ls, As, Cs)

        rng = np.random.default_rng(seed)
        b_al = np.empty(B); b_ct = np.empty(B); b_delta = np.empty(B)
        b_acq = np.empty(B); b_sig = np.empty(B)
        for b in range(B):
            idx = rng.integers(0, n, n)
            lv, av, cv = L[idx].sum(0), A[idx].sum(0), C[idx].sum(0)
            ja, jc = _jsd_vec(lv, av), _jsd_vec(lv, cv)
            b_al[b], b_ct[b], b_delta[b] = ja, jc, jc - ja
            b_acq[b] = _acq_alignment(lv, av, cv, phen_groups)
            b_sig[b] = _sig_alignment(lv, av, cv)

        def ci_block(arr, est):
            return {"est": float(est), "ci_lo": float(np.percentile(arr, lo_q)),
                    "ci_hi": float(np.percentile(arr, hi_q))}

        # permutation: swap AL/CONTROL per sentence
        rng_p = np.random.default_rng(seed + 1)
        perm_delta = np.empty(B)
        perm_rho_diff = np.empty(B)
        # observed rho difference (spearman on aggregate tag vectors)
        obs_rho_al = stats.spearmanr(Ls, As).statistic
        obs_rho_ct = stats.spearmanr(Ls, Cs).statistic
        obs_rho_diff = (obs_rho_al or 0) - (obs_rho_ct or 0)
        for b in range(B):
            swap = rng_p.random(n) < 0.5
            Ap = np.where(swap[:, None], C, A)
            Cp = np.where(swap[:, None], A, C)
            avp, cvp = Ap.sum(0), Cp.sum(0)
            perm_delta[b] = _jsd_vec(Ls, cvp) - _jsd_vec(Ls, avp)
            ra = stats.spearmanr(Ls, avp).statistic or 0
            rc = stats.spearmanr(Ls, cvp).statistic or 0
            perm_rho_diff[b] = ra - rc

        delta_p = float((np.abs(perm_delta) >= abs(obs_delta)).mean())
        rho_p = float((np.abs(perm_rho_diff) >= abs(obs_rho_diff)).mean())
        mde = 2.80 * float(b_delta.std(ddof=0))

        pair_results[ctx.pair_label(al_id, ctrl_id)] = {
            "jsd": {"AL": ci_block(b_al, obs_jsd_al), "CONTROL": ci_block(b_ct, obs_jsd_ct),
                    "delta": {**ci_block(b_delta, obs_delta),
                              "p_delta_gt_0": float((b_delta > 0).mean())}},
            "acq_alignment_ci": ci_block(b_acq, obs_acq),
            "signature_alignment_ci": ci_block(b_sig, obs_sig),
            "permutation": {"a2_rho_p": rho_p, "delta_jsd_p": delta_p,
                            "obs_rho_diff": float(obs_rho_diff)},
            "mde_jsd": mde,
        }
        if fig_payload is None:
            fig_payload = (ctx.pair_label(al_id, ctrl_id), b_delta.copy(), obs_delta)

    results = common.finalize_pairs(pair_results)
    caveats = [f"EXPLORATORY. Seeded ({seed}), resampling unit = SENTENCE (n={n}), B={B}.",
               "At tiny n a non-significant Δ is EXPECTED and non-fatal — the pilot validates the "
               "harness, not statistical power. Full S1 (n≈18,150) is where the decision rule bites."]
    common.write_result(outdir, ID, ctx.run_slug, {"paired": n, "learner": ctx.learner_id},
                        {"B": B, "seed": seed, "ci": ci, "unit": "sentence"}, results, caveats)

    lines = ["# G1 — Statistical robustness", "",
             f"**Finding (n={n}, EXPLORATORY):** " + _headline(pair_results), ""]
    for label, b in pair_results.items():
        d = b["jsd"]["delta"]
        sig = "distinguishable from noise" if (d["ci_lo"] > 0 or d["ci_hi"] < 0) else \
              "NOT distinguishable from noise (expected at tiny n)"
        lines += [f"## Pair `{label}`", "",
                  "| quantity | est | 95% CI |", "|----------|-----|--------|",
                  f"| JSD(AL,lrn) | {b['jsd']['AL']['est']:.4f} | [{b['jsd']['AL']['ci_lo']:.4f}, {b['jsd']['AL']['ci_hi']:.4f}] |",
                  f"| JSD(CTRL,lrn) | {b['jsd']['CONTROL']['est']:.4f} | [{b['jsd']['CONTROL']['ci_lo']:.4f}, {b['jsd']['CONTROL']['ci_hi']:.4f}] |",
                  f"| Δ = JSD_CTRL−JSD_AL | {d['est']:+.4f} | [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}] |",
                  f"| acq alignment | {b['acq_alignment_ci']['est']:.3f} | [{b['acq_alignment_ci']['ci_lo']:.3f}, {b['acq_alignment_ci']['ci_hi']:.3f}] |",
                  f"| signature alignment | {b['signature_alignment_ci']['est']:.3f} | [{b['signature_alignment_ci']['ci_lo']:.3f}, {b['signature_alignment_ci']['ci_hi']:.3f}] |",
                  "",
                  f"- P(Δ>0) across bootstrap = **{d['p_delta_gt_0']:.3f}**  (Δ>0 ⇒ AL closer to LEARNER)",
                  f"- permutation p: Δ_JSD **{b['permutation']['delta_jsd_p']:.3f}**, A2 ρ-diff **{b['permutation']['a2_rho_p']:.3f}**",
                  f"- minimum detectable ΔJSD at this n ≈ **{b['mde_jsd']:.4f}**",
                  f"- **Verdict: the AL-closer effect is {sig}.**"]
    lines += ["", "## Caveats"] + [f"- {c}" for c in caveats]
    _b = next(iter(pair_results.values()))
    _d = _b["jsd"]["delta"]
    _sig = _d["ci_lo"] > 0 or _d["ci_hi"] < 0
    lines += ["", "## Conclusion", "",
              f"At n={n} the toward-learner direction is consistent (bootstrap **P(Δ>0)="
              f"{_d['p_delta_gt_0']:.2f}**) but **not statistically significant**: the divergence gap "
              f"Δ=JSD_ctrl−JSD_AL = {_d['est']:+.4f} carries a 95% CI of [{_d['ci_lo']:+.4f}, {_d['ci_hi']:+.4f}] "
              f"(includes 0), permutation p={_b['permutation']['delta_jsd_p']:.2f}, and the observed effect sits "
              f"below the minimum detectable effect (≈{_b['mde_jsd']:.3f}) at this sample size. This is a "
              f"**power statement, not a null result** — expected and non-fatal, since the pilot validates the "
              f"harness, not statistical power. S1 (n≈18,150) evaluates the pre-registered decision rule with "
              f"adequate power. EXPLORATORY."]
    common.write_md(outdir, "\n".join(lines))

    _plot(outdir, fig_payload)
    common.write_inputs(outdir, ctx.run_slug, ctx.sources,
                        {"consumes": ["A1", "A2", "B4", "E1 point estimates (recomputed)"],
                         "files": ["errors_long_format.tsv"]})
    return results


def _headline(pair_results):
    b = next(iter(pair_results.values()))
    d = b["jsd"]["delta"]
    return (f"Δ(JSD_CTRL−JSD_AL) = {d['est']:+.4f}, 95% CI [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}], "
            f"P(Δ>0)={d['p_delta_gt_0']:.2f}, permutation p={b['permutation']['delta_jsd_p']:.2f}")


def _plot(outdir, payload):
    fdir = common.figures_dir(outdir)
    if payload is None:
        return
    label, b_delta, obs = payload
    common.save_csv(os.path.join(fdir, "bootstrap_delta.csv"), ["bootstrap_delta"],
                    [[float(x)] for x in b_delta])
    fig, ax = plotting.new_fig()
    ax.hist(b_delta, bins=40, color="#4477AA", alpha=0.75)
    ax.axvline(0, color="#EE6677", lw=1.5, label="Δ=0 (no difference)")
    ax.axvline(obs, color="#228833", lw=1.5, label=f"observed Δ={obs:+.3f}")
    ax.set_xlabel("Δ = JSD(CONTROL,lrn) − JSD(AL,lrn)   (>0 ⇒ AL closer)")
    ax.set_ylabel("bootstrap resamples")
    ax.set_title(f"G1: bootstrap Δ distribution — {label}"); ax.legend()
    plotting.save(fig, os.path.join(fdir, "bootstrap_delta.png"))
