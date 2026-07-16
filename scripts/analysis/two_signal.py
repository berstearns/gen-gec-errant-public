"""two_signal — post-hoc Signal-L (length) + Signal-E (error-tag JSD) from pipeline run dirs.

Tier 1 (60-colab-gpu-runbook §Cell 6, 20-metrics-and-stats.md): everything here comes straight from
the saved artifacts — NO pipeline change and NO model reload.

Inputs (per run dir): `raw_results.json` (continuations, prompt_boundaries, perplexities,
region_error_summary per source), `prompts.json` (prompt/reference/full items), `learner_profile.json`
(the authentic-learner yardstick gen-region tag distribution).

Signal L (per generated continuation vs the authentic-learner reference):
  length_ratio = len(gen.split())/len(ref.split());  over_generation_rate (θ=1.5);
  n_sentences via split_into_sentences;  length_distance = |log(length_ratio)|.
  The authentic learner = the reference ⇒ ratio 1 / distance 0 (the yardstick origin).
Signal E: normalized generation-region ERRANT tag distribution per source; JSD-to-learner (base-2,
  common.jsd); RDR = (JSD_control − JSD_AL)/JSD_control; per-acquisition-category ratios.

Emits: signals.json (per-source metrics incl. per-item arrays), matched_pairs.tsv (one row per pair,
both signals), distance_plane.json (per-source point (length_distance, JSD) + per-pair AL−control arrow).

Run:  python -m scripts.analysis.two_signal --run-dirs <dir>... --pair AL:CONTROL --learner-key learner_baseline --out <outdir>
"""
from __future__ import annotations

import argparse
import json
import math
import os
from statistics import median

from gen_gec_errant.preprocessing.runner import split_into_sentences

from . import common

THETA = 1.5  # over-generation threshold: gen_len > ref_len * θ

ACQ_CATS = list(common.ACQUISITION_PHENOMENA.keys())  # sva, verb_morphology, tense, determiner, preposition, noun_number


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def _load_json(path):
    with open(path) as f:
        return json.load(f)


def load_run_dirs(run_dirs):
    """Merge raw_results across run dirs; take prompts + learner_profile from the first dir that has them.
    Returns (raw_by_source, items, learner_profile, provenance)."""
    raw_by_source = {}
    items = None
    learner_profile = None
    seen_dirs = []
    for d in run_dirs:
        rr_path = os.path.join(d, "raw_results.json")
        if not os.path.exists(rr_path):
            raise FileNotFoundError(f"missing raw_results.json in {d}")
        rr = _load_json(rr_path)
        for src, block in rr.items():
            # A source appears identically in every dir that ran it (same corpus+seed); first wins.
            raw_by_source.setdefault(src, block)
        if items is None:
            p_path = os.path.join(d, "prompts.json")
            if os.path.exists(p_path):
                items = _load_json(p_path)
        if learner_profile is None:
            lp_path = os.path.join(d, "learner_profile.json")
            if os.path.exists(lp_path):
                learner_profile = _load_json(lp_path)
        seen_dirs.append(os.path.abspath(d))
    if items is None:
        raise FileNotFoundError("no prompts.json found in any run dir")
    return raw_by_source, items, learner_profile, seen_dirs


# ---------------------------------------------------------------------------
# Signal L
# ---------------------------------------------------------------------------

def signal_l(continuations, references):
    """Per-item length metrics; skips items whose reference has 0 words (ratio undefined)."""
    ratios, distances, nsents, over_flags, multi_flags, ends_term = [], [], [], [], [], []
    per_item = []
    for gen, ref in zip(continuations, references):
        gl = len((gen or "").split())
        rl = len((ref or "").split())
        ns = len(split_into_sentences(gen or ""))
        et = bool(gen) and gen.rstrip().endswith((".", "!", "?"))
        row = {"gen_len": gl, "ref_len": rl, "n_sentences": ns, "ends_with_terminator": et}
        if rl > 0:
            r = gl / rl
            row["length_ratio"] = r
            row["length_distance"] = abs(math.log(r)) if r > 0 else float("inf")
            ratios.append(r)
            if math.isfinite(row["length_distance"]):
                distances.append(row["length_distance"])
            over_flags.append(1 if gl > rl * THETA else 0)
        else:
            row["length_ratio"] = None
            row["length_distance"] = None
        nsents.append(ns)
        multi_flags.append(1 if ns > 1 else 0)
        ends_term.append(1 if et else 0)
        per_item.append(row)

    def _q(xs, p):
        if not xs:
            return None
        s = sorted(xs)
        i = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
        return s[i]

    agg = {
        "n": len(per_item),
        "n_length_scored": len(ratios),
        "median_length_ratio": median(ratios) if ratios else None,
        "length_ratio_iqr": [_q(ratios, 0.25), _q(ratios, 0.75)] if ratios else None,
        "mean_length_distance": (sum(distances) / len(distances)) if distances else None,
        "over_generation_rate": (sum(over_flags) / len(over_flags)) if over_flags else None,
        "mean_n_sentences": (sum(nsents) / len(nsents)) if nsents else None,
        "multi_sentence_rate": (sum(multi_flags) / len(multi_flags)) if multi_flags else None,
        "ends_with_terminator_rate": (sum(ends_term) / len(ends_term)) if ends_term else None,
    }
    return agg, per_item


# ---------------------------------------------------------------------------
# Signal E
# ---------------------------------------------------------------------------

def gen_region_counts(block):
    """generation-region ERRANT tag counts for one raw_results source block."""
    res = block.get("region_error_summary", {})
    return dict(res.get("generation_error_type_counts", {}))


def acquisition_counts(tag_counts):
    """Sum tag counts into the six acquisition phenomena (CONCEPTS.md §1.2 / B4)."""
    out = {}
    for cat, tags in common.ACQUISITION_PHENOMENA.items():
        out[cat] = sum(tag_counts.get(t, 0) for t in tags)
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def compute(run_dirs, pair, learner_key):
    raw, items, learner_profile, seen_dirs = load_run_dirs(run_dirs)
    references = [it.get("reference", "") for it in items]

    al_key, ctrl_key = pair
    for k in (al_key, ctrl_key, learner_key):
        if k not in raw:
            raise KeyError(f"source '{k}' not in raw_results (have: {sorted(raw)})")

    role_of = {learner_key: "authentic_learner", ctrl_key: "matched_control", al_key: "artificial_learner"}

    # yardstick gen-region tag distribution: prefer learner_profile.json, else learner_baseline block
    if learner_profile and learner_profile.get("generation_region_error_type_counts"):
        learner_counts = dict(learner_profile["generation_region_error_type_counts"])
    else:
        learner_counts = gen_region_counts(raw[learner_key])

    sources = {}
    for src in (learner_key, ctrl_key, al_key):
        block = raw[src]
        conts = block.get("continuations", [])
        l_agg, l_items = signal_l(conts, references)
        tag_counts = gen_region_counts(block)
        n_items = l_agg["n"] or len(conts)
        ppls = [p for p in block.get("perplexities", []) if p]  # learner dummy 0.0 dropped
        is_learner = src == learner_key
        jsd = 0.0 if is_learner else common.jsd_counts(tag_counts, learner_counts)
        gen_total = block.get("region_error_summary", {}).get("generation_total_errors",
                                                              sum(tag_counts.values()))
        sources[src] = {
            "id": src,
            "role": role_of[src],
            "length": l_agg,
            "length_per_item": l_items,
            "gen_tag_counts": tag_counts,
            "gen_tag_shares": common.shares(tag_counts),
            "acquisition_counts": acquisition_counts(tag_counts),
            "jsd_to_learner": jsd,
            "generation_total_errors": gen_total,
            "errors_per_sentence": (gen_total / n_items) if n_items else None,
            "mean_perplexity": (sum(ppls) / len(ppls)) if ppls else None,
        }

    # RDR + arrow
    jsd_al = sources[al_key]["jsd_to_learner"]
    jsd_ctrl = sources[ctrl_key]["jsd_to_learner"]
    rdr = (jsd_ctrl - jsd_al) / jsd_ctrl if jsd_ctrl > 0 else None
    ld_al = sources[al_key]["length"]["mean_length_distance"] or 0.0
    ld_ctrl = sources[ctrl_key]["length"]["mean_length_distance"] or 0.0

    learner_acq = acquisition_counts(learner_counts)
    acq_ratio = {}
    for cat in ACQ_CATS:
        la = learner_acq.get(cat, 0)
        al_c = sources[al_key]["acquisition_counts"].get(cat, 0)
        ct_c = sources[ctrl_key]["acquisition_counts"].get(cat, 0)
        acq_ratio[cat] = {
            "learner": la, "al": al_c, "control": ct_c,
            "al_over_control": (al_c / ct_c) if ct_c else (float("inf") if al_c else None),
        }

    joint_inward = (ld_al < ld_ctrl) and (jsd_al < jsd_ctrl)  # both closer to the yardstick

    pair_block = {
        "label": f"{al_key}:{ctrl_key}",
        "al": al_key, "control": ctrl_key,
        "length_ratio_al": sources[al_key]["length"]["median_length_ratio"],
        "length_ratio_control": sources[ctrl_key]["length"]["median_length_ratio"],
        "length_distance_al": ld_al,
        "length_distance_control": ld_ctrl,
        "delta_length_distance": ld_al - ld_ctrl,
        "jsd_al": jsd_al, "jsd_control": jsd_ctrl, "delta_jsd": jsd_al - jsd_ctrl,
        "rdr": rdr,
        "acquisition_ratios": acq_ratio,
        "errors_per_sentence_al": sources[al_key]["errors_per_sentence"],
        "errors_per_sentence_control": sources[ctrl_key]["errors_per_sentence"],
        "ppl_al": sources[al_key]["mean_perplexity"],
        "ppl_control": sources[ctrl_key]["mean_perplexity"],
        "joint_inward": joint_inward,
    }

    distance_plane = {
        "axes": {"x": "length_distance = |log(gen_len/ref_len)|", "y": "JSD(source, authentic learner)"},
        "origin": {"role": "authentic_learner", "point": [0.0, 0.0]},
        "points": {
            role_of[src]: {
                "id": src,
                "length_distance": sources[src]["length"]["mean_length_distance"],
                "jsd": sources[src]["jsd_to_learner"],
            }
            for src in (learner_key, ctrl_key, al_key)
        },
        "arrows": [{
            "pair": pair_block["label"],
            "from_role": "matched_control", "to_role": "artificial_learner",
            "d_length_distance": ld_al - ld_ctrl,
            "d_jsd": jsd_al - jsd_ctrl,
            "inward_both": joint_inward,
        }],
    }

    signals = {
        "provenance": {
            "run_dirs": seen_dirs, "pair": list(pair), "learner_key": learner_key,
            "n_items": len(items), "theta_over_generation": THETA,
        },
        "learner_gen_tag_counts": learner_counts,
        "sources": sources,
        "pairs": [pair_block],
    }
    return signals, distance_plane, [pair_block]


def write_outputs(outdir, signals, distance_plane, pairs):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "signals.json"), "w") as f:
        json.dump(signals, f, indent=2, default=str)
    with open(os.path.join(outdir, "distance_plane.json"), "w") as f:
        json.dump(distance_plane, f, indent=2, default=str)

    header = [
        "pair", "length_ratio_al", "length_ratio_control", "delta_length_distance",
        "jsd_al", "jsd_control", "rdr",
        "acq_sva_al_over_control", "acq_verb_morphology_al_over_control", "acq_determiner_al_over_control",
        "errors_per_sentence_al", "errors_per_sentence_control", "ppl_al", "ppl_control", "joint_inward",
    ]
    rows = []
    for p in pairs:
        ar = p["acquisition_ratios"]
        rows.append([
            p["label"], p["length_ratio_al"], p["length_ratio_control"], p["delta_length_distance"],
            p["jsd_al"], p["jsd_control"], p["rdr"],
            ar["sva"]["al_over_control"], ar["verb_morphology"]["al_over_control"], ar["determiner"]["al_over_control"],
            p["errors_per_sentence_al"], p["errors_per_sentence_control"], p["ppl_al"], p["ppl_control"], p["joint_inward"],
        ])
    common.save_csv(os.path.join(outdir, "matched_pairs.tsv"), header, rows)
    # save_csv writes comma-delimited; rewrite tab-delimited for .tsv
    with open(os.path.join(outdir, "matched_pairs.tsv"), "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join("" if v is None else str(v) for v in r) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Two-signal (length + error-tag JSD) post-hoc analysis")
    ap.add_argument("--run-dirs", nargs="+", required=True, help="pipeline run dirs with raw_results.json")
    ap.add_argument("--pair", required=True, help="AL_key:CONTROL_key (raw_results model names)")
    ap.add_argument("--learner-key", default="learner_baseline")
    ap.add_argument("--out", required=True, help="output dir for signals/tsv/json")
    args = ap.parse_args(argv)

    al_key, ctrl_key = args.pair.split(":", 1)
    signals, plane, pairs = compute(args.run_dirs, (al_key, ctrl_key), args.learner_key)
    write_outputs(args.out, signals, plane, pairs)

    p = pairs[0]
    print(f"[two_signal] pair {p['label']}")
    print(f"  JSD  AL={p['jsd_al']:.4f}  control={p['jsd_control']:.4f}  RDR={p['rdr']}")
    print(f"  len-dist  AL={p['length_distance_al']:.4f}  control={p['length_distance_control']:.4f}  Δ={p['delta_length_distance']:.4f}")
    print(f"  median length_ratio  AL={p['length_ratio_al']}  control={p['length_ratio_control']}")
    print(f"  joint inward (both toward yardstick): {p['joint_inward']}")
    print(f"  wrote: {args.out}/{{signals.json,distance_plane.json,matched_pairs.tsv}}")
    print("TWO_SIGNAL_OK")


if __name__ == "__main__":
    main()
