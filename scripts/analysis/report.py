"""REPORT.md — auto-assembled digest across all analyses. Regenerated, never
hand-edited. Reads each analysis's result.json (never re-derives)."""
from __future__ import annotations

import json
import os


def _load(out_root, slug):
    p = os.path.join(out_root, slug, "result.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def _pairs_of(res):
    if res and "results" in res and "by_pair" in res["results"]:
        return res["results"]["by_pair"]
    return {}


def assemble(ctx, manifest) -> str:
    R = {aid: _load(ctx.out_root, slug) for aid, slug in [
        ("A1", "A1-distributional-similarity"), ("A2", "A2-rank-correlation-displacement"),
        ("B1", "B1-operation-mru"), ("B2", "B2-pos-family"), ("B3", "B3-finegrained-tag"),
        ("B4", "B4-acquisition-categories"), ("C1", "C1-magnitude-density"),
        ("D1", "D1-region-prompt-vs-generation"), ("D2", "D2-length-fluency-confounds"),
        ("D3", "D3-degeneracy-screen"), ("D4", "D4-gec-artifact-screen"),
        ("E1", "E1-overrepresentation-signature"), ("G1", "G1-statistical-robustness"),
        ("Z1", "Z1-findings-synthesis")]}

    n = ctx.paired["n"]
    L = []
    L += [f"# REPORT — {ctx.run_slug}", "",
          "> **EXPLORATORY, NOT CONFIRMATORY.** This run validates the analysis harness and "
          f"generates hypotheses on a tiny paired sample (n={n}). The confirmatory evidence is the "
          "full-S1 run (see gen-gec-review-specs/20-preregistration.md). REPORT.md is auto-assembled.",
          ""]

    # 1. FINDINGS FIRST (Z1) — conclusions before numbers
    L += _findings_lead(R["Z1"], ctx)

    L += ["## 2. Run header", "",
          f"- run-slug: `{ctx.run_slug}`",
          f"- paired n: **{n}** sentences (shared text_id×sentence_idx across LEARNER/AL/CONTROL)",
          f"- learner: `{ctx.learner_id}` · pairs: {', '.join(f'`{a}:{c}`' for a, c in ctx.pairs)}",
          f"- code git sha: `{manifest['code_git_sha'][:12]}` · created: {manifest['created_ts']}",
          f"- instrument: GEC=`{manifest['instrument'].get('gec_model')}`, "
          f"spaCy={manifest['instrument'].get('spacy_model')}, split_sentences="
          f"{manifest['instrument'].get('split_sentences')}",
          f"- inputs (hashed in MANIFEST.json): "
          + ", ".join(f"{i['id']}→`{os.path.basename(i['run_dir'])}`" for i in manifest["inputs"]),
          ""]

    # 3. headline table
    L += ["## 3. Headline metrics — is AL closer to LEARNER than CONTROL?", "",
          "| pair | JSD_tag(AL,lrn) | JSD_tag(CTRL,lrn) | AL closer? | Δ 95% CI (G1) | P(Δ>0) | perm p |",
          "|------|-----------------|-------------------|-----------|---------------|--------|--------|"]
    a1 = _pairs_of(R["A1"]); g1 = _pairs_of(R["G1"])
    for a, c in ctx.pairs:
        label = f"{a}:{c}"
        jrow = "—"; g = ""
        if label in a1:
            tag = a1[label]["tag"]
            jal, jct = tag["AL"]["jsd"], tag["CONTROL"]["jsd"]
            closer = "**yes**" if tag["AL_closer"] else "no"
            jrow = f"| `{label}` | {jal:.4f} | {jct:.4f} | {closer} |"
        gci = "—"; pdg = "—"; pp = "—"
        if label in g1:
            d = g1[label]["jsd"]["delta"]
            gci = f"[{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]"; pdg = f"{d['p_delta_gt_0']:.2f}"
            pp = f"{g1[label]['permutation']['delta_jsd_p']:.2f}"
        L.append(f"{jrow} {gci} | {pdg} | {pp} |")
    L.append("")

    # 4. per-RQ verdicts
    L += ["## 4. Per-RQ verdicts", ""]
    L += _rq_alpha(R) + _rq_beta(R) + _rq_gamma(R) + _rq_delta(R) + _rq_eps(R) + _rq_zeta(R)

    # 5. caveats union
    L += ["", "## 5. Open caveats (union across analyses)", ""]
    seen = set()
    for aid, res in R.items():
        if not res:
            continue
        for cv in res.get("caveats", []):
            key = cv.strip()
            if key not in seen:
                seen.add(key)
                L.append(f"- [{aid}] {cv}")

    # 6. what changes at full S1
    L += ["", "## 6. What changes at full S1", "",
          "- Same driver, same output tree — only `--run-slug s1-full-<date>` and a 5-pair "
          "`--pairs` roster change (H1). Per-tag panels (B3/B4) and G1 CIs gain real power at "
          "n≈18,150; the pre-registered decision rule (A1 JSD + B4 ≥2/4 alignment) is evaluated "
          "on G1's CIs there, not on these point estimates.",
          "- The tiny-sample numbers above are hypothesis-generating only.", ""]
    return "\n".join(L)


def _findings_lead(z1, ctx):
    """Section 1: the Z1 findings, conclusions-first (claim + effect + confidence)."""
    out = ["## 1. Findings (Z1 synthesis — read this first)", "",
           "> Full deliverables: `Z1-findings-synthesis/FINDINGS.md`, `results-narrative.md` "
           "(paper-ready), `findings.json`. Each finding carries a named effect size and G1-tied confidence.",
           ""]
    if not z1:
        return out + ["_Z1 did not run._", ""]
    by_pair = z1["results"].get("by_pair", {})
    for label, findings in by_pair.items():
        out.append(f"**Pair `{label}`**")
        out += ["", "| # | finding | effect size | direction | confidence |",
                "|---|---------|-------------|-----------|------------|"]
        for F in findings:
            es = F["effect_size"]
            val = es["value"]
            val_s = f"{val:.3f}" if isinstance(val, float) else str(val)
            if len(val_s) > 40:
                val_s = val_s[:38] + "…"
            out.append(f"| {F['id']} | {F['title']} | {es['measure']} = {val_s} | {F['direction']} | "
                       f"{F['confidence']['level']} |")
        # headline claim spelled out
        head = findings[0]
        out += ["", f"**Headline claim.** {head['claim']}", ""]
    return out


def _cite(R, aid):
    slug = {"A1": "A1-distributional-similarity", "A2": "A2-rank-correlation-displacement",
            "B1": "B1-operation-mru", "B2": "B2-pos-family", "B4": "B4-acquisition-categories",
            "C1": "C1-magnitude-density", "D1": "D1-region-prompt-vs-generation",
            "D2": "D2-length-fluency-confounds", "D3": "D3-degeneracy-screen",
            "D4": "D4-gec-artifact-screen", "E1": "E1-overrepresentation-signature",
            "G1": "G1-statistical-robustness"}[aid]
    return f"`{slug}/result.json`"


def _first(pairs):
    return next(iter(pairs.values())) if pairs else None


def _rq_alpha(R):
    a1 = _first(_pairs_of(R["A1"])); a2 = _first(_pairs_of(R["A2"])); e1 = _first(_pairs_of(R["E1"]))
    out = ["### α — similarity (A1, A2, E1)"]
    if a1:
        out.append(f"- A1 {_cite(R,'A1')}: JSD_tag AL={a1['tag']['AL']['jsd']:.3f} vs "
                   f"CONTROL={a1['tag']['CONTROL']['jsd']:.3f} → AL "
                   f"{'closer' if a1['tag']['AL_closer'] else 'not closer'}.")
    if a2:
        out.append(f"- A2 {_cite(R,'A2')}: τ-b AL={a2['kendall']['AL']:.3f} vs CONTROL={a2['kendall']['CONTROL']:.3f}.")
    if e1:
        s = e1["signature_alignment"]
        out.append(f"- E1 {_cite(R,'E1')}: signature alignment {s['n_toward']}/{s['n_total']} learner-present tags toward LEARNER.")
    return out + [""]


def _rq_beta(R):
    b1 = _first(_pairs_of(R["B1"])); b2 = _first(_pairs_of(R["B2"])); b4 = _first(_pairs_of(R["B4"]))
    out = ["### β — structure (B1, B2, B3, B4)"]
    if b1:
        out.append(f"- B1 {_cite(R,'B1')}: operation-balance AL closer to LEARNER? {b1['AL_closer']} "
                   f"(omission M:(R+U) LEARNER={_om(b1['LEARNER'])}, AL={_om(b1['AL'])}, CONTROL={_om(b1['CONTROL'])}).")
    if b2:
        out.append(f"- B2 {_cite(R,'B2')}: POS directional agreement = {b2['directional_agreement']:.2f}.")
    if b4:
        a = b4["acquisition_alignment"]
        out.append(f"- B4 {_cite(R,'B4')}: acquisition alignment **{a['n_toward_learner']}/{a['n_total']}** "
                   f"phenomena toward LEARNER (pre-registered ≥2/4 companion).")
    return out + [""]


def _om(p):
    return f"{p['omission_ratio']:.2f}" if p.get("omission_ratio") is not None else "—"


def _rq_gamma(R):
    c1 = _first(_pairs_of(R["C1"]))
    out = ["### γ — magnitude (C1)"]
    if c1:
        out.append(f"- C1 {_cite(R,'C1')}: errs/sent LEARNER={c1['LEARNER']['err_per_sent']['mean']:.2f}, "
                   f"AL={c1['AL']['err_per_sent']['mean']:.2f}, CONTROL={c1['CONTROL']['err_per_sent']['mean']:.2f}; "
                   f"density LEARNER={c1['LEARNER']['density']['mean']:.3f}, AL={c1['AL']['density']['mean']:.3f}, "
                   f"CONTROL={c1['CONTROL']['density']['mean']:.3f}.")
    return out + [""]


def _rq_delta(R):
    d1 = R["D1"]; d2 = _first(_pairs_of(R["D2"])); d3 = R["D3"]; d4 = _first(_pairs_of(R["D4"]))
    out = ["### δ — validity (D1–D4)"]
    if d1:
        bi = d1["results"]["boundary_integrity"]
        pd = d1["results"]["prompt_drift"]
        out.append(f"- D1 {_cite(R,'D1')}: boundary-integrity gate **{'PASS' if bi['ok'] else 'FAIL'}** "
                   f"(span alignment + count reconciliation). Prompt-drift TVD={pd['max_pairwise_tvd']:.3f} "
                   "(ADVISORY — joint-correction, does not gate or provision gen-region claims).")
    if d3:
        cis = d3["results"]["clean_index_size"]
        out.append(f"- D3 {_cite(R,'D3')}: clean index {cis}/{d3['n']['paired']} (degeneracy screen).")
    if d2:
        out.append(f"- D2 {_cite(R,'D2')}: AL-closer ranking survives length-matching? {d2['ranking_holds']}.")
    if d4:
        out.append(f"- D4 {_cite(R,'D4')}: AL-closer ranking survives artifact removal? {d4['ranking_holds']} "
                   f"(content-only JSD AL={d4['content_only_jsd']['AL']:.3f} vs CONTROL={d4['content_only_jsd']['CONTROL']:.3f}).")
    return out + [""]


def _rq_eps(R):
    g1 = _first(_pairs_of(R["G1"]))
    out = ["### ε — robustness (G1)"]
    if g1:
        d = g1["jsd"]["delta"]
        sig = d["ci_lo"] > 0 or d["ci_hi"] < 0
        out.append(f"- G1 {_cite(R,'G1')}: Δ={d['est']:+.4f} 95% CI [{d['ci_lo']:+.4f},{d['ci_hi']:+.4f}], "
                   f"P(Δ>0)={d['p_delta_gt_0']:.2f}, perm p={g1['permutation']['delta_jsd_p']:.2f}, "
                   f"MDE≈{g1['mde_jsd']:.4f} → effect **{'distinguishable' if sig else 'NOT distinguishable'}** "
                   "from noise (non-significance is expected & non-fatal at tiny n).")
    return out + [""]


def _rq_zeta(R):
    return ["### ζ — qualitative (F1)",
            "- F1 `F1-qualitative-paired-examples/result.json`: curated convergent / AL-only / "
            "learner-only / artifact example banks (see result.md + examples_for_paper.md).", ""]


def write_report(out_root: str, text: str) -> None:
    with open(os.path.join(out_root, "REPORT.md"), "w") as f:
        f.write(text.rstrip() + "\n")
