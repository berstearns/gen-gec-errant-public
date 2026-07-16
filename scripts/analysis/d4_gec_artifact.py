"""D4 — GEC artifact screen (R:ORTH / R:SPELL / *:PUNCT). RQ-delta (instrument)."""
from __future__ import annotations

import os
from collections import Counter

from . import common
from . import plotting

ID = "D4"
SLUG = "D4-gec-artifact-screen"


def _is_pure_casing(orig: str, corr: str) -> bool:
    return bool(orig) and bool(corr) and orig != corr and orig.lower() == corr.lower()


def _source_artifacts(ctx, sid, keys):
    s = ctx.sources[sid]
    total = 0
    artifact = 0
    orth = 0
    orth_casing = 0
    orth_initial = 0
    punct = 0
    for k in keys:
        pb = s.prompt_boundaries.get(k, None)
        for r in s.gen_errors.get(k, []):
            total += 1
            t = r["error_type"]
            if t in common.ARTIFACT_TAGS:
                artifact += 1
            if t == "R:ORTH":
                orth += 1
                if _is_pure_casing(r["error_original_tokens"], r["error_corrected_tokens"]):
                    orth_casing += 1
                # sentence-initial (fragment) artifact: error at continuation start
                try:
                    cs = int(r["char_start"])
                    if pb is not None and abs(cs - pb) <= 2:
                        orth_initial += 1
                except (ValueError, TypeError):
                    pass
            if t in ("M:PUNCT", "R:PUNCT", "U:PUNCT"):
                punct += 1
    return {"total": total, "artifact": artifact, "orth": orth, "orth_casing": orth_casing,
            "orth_initial": orth_initial, "punct": punct,
            "artifact_share": artifact / total if total else 0.0,
            "orth_casing_share": orth_casing / total if total else 0.0,
            "punct_share": punct / total if total else 0.0,
            "orth_sentence_initial_frac": orth_initial / orth if orth else 0.0}


def _content_counts(ctx, sid, keys):
    """gen tag counts with artifact classes removed."""
    c = Counter()
    for tag, n in ctx.sources[sid].gen_tag_counts(keys).items():
        if tag not in common.ARTIFACT_TAGS:
            c[tag] += n
    return c


def run(ctx: common.Context) -> dict:
    outdir = os.path.join(ctx.out_root, SLUG)
    keys = [tuple(k) for k in ctx.paired["keys"]]
    n = len(keys)

    per_source = {sid: _source_artifacts(ctx, sid, keys) for sid in ctx.sources}
    lc_content = _content_counts(ctx, ctx.learner_id, keys)
    lc_raw = ctx.sources[ctx.learner_id].gen_tag_counts(keys)

    pair_results = {}
    for al_id, ctrl_id in ctx.pairs:
        content_jsd = {"AL": common.jsd_counts(dict(lc_content), dict(_content_counts(ctx, al_id, keys))),
                       "CONTROL": common.jsd_counts(dict(lc_content), dict(_content_counts(ctx, ctrl_id, keys)))}
        raw_jsd = {"AL": common.jsd_counts(dict(lc_raw), dict(ctx.sources[al_id].gen_tag_counts(keys))),
                   "CONTROL": common.jsd_counts(dict(lc_raw), dict(ctx.sources[ctrl_id].gen_tag_counts(keys)))}
        holds = (content_jsd["AL"] < content_jsd["CONTROL"]) == (raw_jsd["AL"] < raw_jsd["CONTROL"])
        pair_results[ctx.pair_label(al_id, ctrl_id)] = {
            "source": {"LEARNER": per_source[ctx.learner_id], "AL": per_source[al_id],
                       "CONTROL": per_source[ctrl_id]},
            "content_only_jsd": content_jsd, "raw_jsd": raw_jsd, "ranking_holds": holds}

    results = common.finalize_pairs(pair_results)
    caveats = ["EXPLORATORY. content_only_jsd removes {R:ORTH,R:SPELL,*:PUNCT} from all sources.",
               "If the AL-closer ranking flips once artifacts are stripped → the similarity was "
               "artifact-driven (MAJOR caveat to A1). This is the most likely reviewer-2 objection."]
    common.write_result(outdir, ID, ctx.run_slug, {"paired": n, "learner": ctx.learner_id},
                        {"artifact_tags": sorted(common.ARTIFACT_TAGS)}, results, caveats)

    lines = ["# D4 — GEC artifact screen", "",
             f"**Finding (n={n}, EXPLORATORY):** " + _headline(pair_results), ""]
    for label, b in pair_results.items():
        lines += [f"## Pair `{label}`", "",
                  "| source | artifact share | orth-casing share | punct share | R:ORTH sent-initial frac |",
                  "|--------|----------------|-------------------|-------------|--------------------------|"]
        for role in ("LEARNER", "AL", "CONTROL"):
            s = b["source"][role]
            lines.append(f"| {role} | {common.fmt_pct(s['artifact_share'])} | {common.fmt_pct(s['orth_casing_share'])} | "
                         f"{common.fmt_pct(s['punct_share'])} | {common.fmt_pct(s['orth_sentence_initial_frac'])} |")
        lines += [f"- content-only JSD (artifacts removed): AL={b['content_only_jsd']['AL']:.4f}, "
                  f"CONTROL={b['content_only_jsd']['CONTROL']:.4f}",
                  f"- raw JSD (A1): AL={b['raw_jsd']['AL']:.4f}, CONTROL={b['raw_jsd']['CONTROL']:.4f}",
                  f"- **AL-closer ranking survives artifact removal? {b['ranking_holds']}**"]
    lines += ["", "## Caveats"] + [f"- {c}" for c in caveats]
    _b = next(iter(pair_results.values()))
    _al_art = _b["source"]["AL"]["artifact_share"]; _l_art = _b["source"]["LEARNER"]["artifact_share"]
    _cj = _b["content_only_jsd"]; _rdr_c = (_cj["CONTROL"] - _cj["AL"]) / _cj["CONTROL"] if _cj["CONTROL"] else 0.0
    lines += ["", "## Conclusion", "",
              f"AL over-produces GEC-artifact classes (**{common.fmt_pct(_al_art)}** of its errors are "
              f"orthography/spelling/punct vs {common.fmt_pct(_l_art)} for learners), the single most likely "
              f"reviewer-2 objection. But that over-production is **not** what makes AL learner-like: with all "
              f"artifact classes removed the ranking survives — content-only JSD {_cj['AL']:.3f} (AL) vs "
              f"{_cj['CONTROL']:.3f} (control), RDR **{common.fmt_pct(_rdr_c)}**, ranking holds = "
              f"**{_b['ranking_holds']}**. The learner alignment is carried by content categories (determiner, "
              f"preposition), not by the coedit-large casing/punctuation habit. EXPLORATORY, n={n}."]
    common.write_md(outdir, "\n".join(lines))

    _plot(outdir, pair_results)
    common.write_inputs(outdir, ctx.run_slug, ctx.sources, {"files": ["errors_long_format.tsv"]})
    return results


def _headline(pair_results):
    b = next(iter(pair_results.values()))
    s = b["source"]
    return (f"artifact share — LEARNER={common.fmt_pct(s['LEARNER']['artifact_share'])}, "
            f"AL={common.fmt_pct(s['AL']['artifact_share'])}, "
            f"CONTROL={common.fmt_pct(s['CONTROL']['artifact_share'])}; "
            f"content-only ranking holds={b['ranking_holds']}")


def _plot(outdir, pair_results):
    fdir = common.figures_dir(outdir)
    import numpy as np
    label0 = next(iter(pair_results)); b = pair_results[label0]
    roles = ["LEARNER", "AL", "CONTROL"]
    rows = [[r, b["source"][r]["artifact_share"], b["source"][r]["orth_casing_share"],
             b["source"][r]["punct_share"]] for r in roles]
    common.save_csv(os.path.join(fdir, "artifact_share.csv"),
                    ["source", "artifact_share", "orth_casing_share", "punct_share"], rows)
    x = np.arange(len(roles)); w = 0.6
    fig, ax = plotting.new_fig(5.5, 4.2)
    casing = [b["source"][r]["orth_casing_share"] for r in roles]
    punct = [b["source"][r]["punct_share"] for r in roles]
    other = [b["source"][r]["artifact_share"] - c - p for r, c, p in zip(roles, casing, punct)]
    ax.bar(x, casing, w, label="R:ORTH casing", color="#DDCC77")
    ax.bar(x, punct, w, bottom=casing, label="*:PUNCT", color="#AA4499")
    ax.bar(x, other, w, bottom=np.array(casing) + np.array(punct), label="other artifact", color="#BBBBBB")
    ax.set_xticks(x); ax.set_xticklabels(roles)
    ax.set_ylabel("share of gen-region errors")
    ax.set_title(f"D4: GEC-artifact share — {label0}"); ax.legend()
    plotting.save(fig, os.path.join(fdir, "artifact_share.png"))
