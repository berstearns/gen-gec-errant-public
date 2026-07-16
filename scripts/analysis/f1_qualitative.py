"""F1 — Qualitative paired examples. RQ-zeta. Deterministic (sort by key)."""
from __future__ import annotations

import os

from . import common

ID = "F1"
SLUG = "F1-qualitative-paired-examples"


def _edits(rows):
    return [{"type": r["error_type"], "orig": r["error_original_tokens"],
             "corr": r["error_corrected_tokens"]} for r in rows]


def _tags(rows):
    return sorted({r["error_type"] for r in rows})


def _example(prompts, learner, al, ctrl, k):
    return {
        "sentence_id": f"{k[0]}:{k[1]}",
        "prompt": prompts.get(k, ""),
        "learner": {"cont": learner.continuations.get(k, ""), "tags": _tags(learner.gen_errors.get(k, [])),
                    "edits": _edits(learner.gen_errors.get(k, []))},
        "AL": {"cont": al.continuations.get(k, ""), "tags": _tags(al.gen_errors.get(k, [])),
               "edits": _edits(al.gen_errors.get(k, []))},
        "CONTROL": {"cont": ctrl.continuations.get(k, ""), "tags": _tags(ctrl.gen_errors.get(k, [])),
                    "edits": _edits(ctrl.gen_errors.get(k, []))},
    }


def run(ctx: common.Context) -> dict:
    outdir = os.path.join(ctx.out_root, SLUG)
    keys = [tuple(k) for k in ctx.paired["keys"]]
    n = len(keys)
    n_per = ctx.params.get("n_per_bucket", 5)
    prompts = common.load_prompts(ctx.sources[ctx.learner_id].run_dir)
    acq = set(common.ACQUISITION_TAGS)
    artifact = common.ARTIFACT_TAGS

    learner = ctx.sources[ctx.learner_id]
    pair_results = {}
    for al_id, ctrl_id in ctx.pairs:
        al = ctx.sources[al_id]; ctrl = ctx.sources[ctrl_id]
        buckets = {"convergent": [], "al_only": [], "learner_only": [], "artifact": []}
        for k in keys:  # keys already sorted deterministically
            lt = set(_tags(learner.gen_errors.get(k, [])))
            at = set(_tags(al.gen_errors.get(k, [])))
            ct = set(_tags(ctrl.gen_errors.get(k, [])))
            # 1. convergent: shared acquisition tag AL∩LEARNER, absent from CONTROL
            conv = (lt & at & acq) - ct
            if conv and len(buckets["convergent"]) < n_per:
                ex = _example(prompts, learner, al, ctrl, k)
                ex["shared_tags"] = sorted(conv)
                buckets["convergent"].append(ex)
            # 2. al_only: AL tag absent from LEARNER
            if (at - lt) and len(buckets["al_only"]) < n_per:
                ex = _example(prompts, learner, al, ctrl, k)
                ex["al_only_tags"] = sorted(at - lt)
                buckets["al_only"].append(ex)
            # 3. learner_only: LEARNER tag AL misses
            if (lt - at) and len(buckets["learner_only"]) < n_per:
                ex = _example(prompts, learner, al, ctrl, k)
                ex["learner_only_tags"] = sorted(lt - at)
                buckets["learner_only"].append(ex)
            # 4. artifact: any artifact tag present in AL
            if (at & artifact) and len(buckets["artifact"]) < n_per:
                ex = _example(prompts, learner, al, ctrl, k)
                ex["artifact_tags"] = sorted(at & artifact)
                buckets["artifact"].append(ex)
        pair_results[ctx.pair_label(al_id, ctrl_id)] = buckets

    results = common.finalize_pairs(pair_results)
    caveats = ["EXPLORATORY. Deterministic selection (sort by sentence_id, first n per bucket) — same "
               "examples every re-run. Every tag traces to an errors_long row.",
               "Convergent bucket (AL & LEARNER share an acquisition tag CONTROL lacks) is the paper's "
               "best illustrative story."]
    common.write_result(outdir, ID, ctx.run_slug, {"paired": n, "learner": ctx.learner_id},
                        {"n_per_bucket": n_per}, results, caveats)

    lines = ["# F1 — Qualitative paired examples", "",
             f"**(n={n}, EXPLORATORY)** Same-prompt continuations across LEARNER / AL / CONTROL.", ""]
    paper_lines = ["# Paper-ready paired examples (F1)", ""]
    for label, buckets in pair_results.items():
        lines.append(f"## Pair `{label}`")
        for bname, exs in buckets.items():
            lines += ["", f"### {bname}  ({len(exs)})"]
            for ex in exs:
                lines += _render(ex, bname)
        # convergent → paper
        for ex in buckets["convergent"]:
            paper_lines += _render(ex, "convergent") + [""]
    lines += ["", "## Caveats"] + [f"- {c}" for c in caveats]
    _b = next(iter(pair_results.values()))
    _nc = len(_b["convergent"]); _na = len(_b["al_only"]); _nl = len(_b["learner_only"]); _nf = len(_b["artifact"])
    lines += ["", "## Conclusion", "",
              f"On shared prompts the harness surfaces **{_nc}** convergent case(s) where AL and the authentic "
              f"learner commit the *same acquisition-tag error* (e.g. R:VERB:SVA) that the control does not — "
              f"concrete grounding for the quantitative alignment (A1/B4) and the paper's illustrative-examples "
              f"section. It also isolates {_na} AL-only (hallucinated), {_nl} learner-only (coverage-gap), and "
              f"{_nf} artifact case(s), giving reviewers auditable instances of every failure mode. Selection "
              f"is deterministic (sort by sentence_id); every tag traces to an errors_long row. EXPLORATORY, "
              f"n={n}."]
    common.write_md(outdir, "\n".join(lines))
    with open(os.path.join(outdir, "examples_for_paper.md"), "w") as f:
        f.write("\n".join(paper_lines).rstrip() + "\n")

    common.figures_dir(outdir)  # keep contract shape (empty figures/ ok for F1)
    common.write_inputs(outdir, ctx.run_slug, ctx.sources,
                        {"files": ["full_results.tsv", "errors_long_format.tsv", "raw_results.json"]})
    return results


def _render(ex, bname):
    lines = [f"**{ex['sentence_id']}** — prompt: _{_clip(ex['prompt'])}_"]
    key = {"convergent": "shared_tags", "al_only": "al_only_tags",
           "learner_only": "learner_only_tags", "artifact": "artifact_tags"}.get(bname)
    if key and ex.get(key):
        lines.append(f"- focus tags: `{', '.join(ex[key])}`")
    for role in ("learner", "AL", "CONTROL"):
        r = ex[role]
        edits = "; ".join(f"{e['orig']!r}→{e['corr']!r} [{e['type']}]" for e in r["edits"]) or "—"
        lines.append(f"- **{role}**: _{_clip(r['cont'])}_  · edits: {edits}")
    return lines


def _clip(s, n=160):
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[:n] + "…"
