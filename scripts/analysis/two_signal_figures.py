"""two_signal_figures — Figs 1-4 + Tables 1-2 from a two_signal.py output dir (30-figures-and-tables.md).

Canonical everywhere (30-figures §Conventions):
  source order  = authentic learner · matched control · artificial learner (AL)
  palette       = authentic #d1495b (yardstick, warm) · control #8d99ae (muted) · AL #2a9d8f (teal)
  yardstick     = origin / reference line, never "just another bar".

FIG 1 length over-generation (box+points, ref line at ratio=1) · TABLE 1 length metrics ·
FIG 2 error-tag distance (per-source tag bars + acquisition panel, JSD annotated) ·
FIG 3 the 2-D distance-to-learner plane (THE money figure; control→AL arrow) ·
FIG 4 per-criterion movement (length + JSD with RDR) · TABLE 2 matched-pair results.
Every figure saved PNG + SVG.

Run:  python -m scripts.analysis.two_signal_figures --in <two_signal_out_dir> [--out <dir>]
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ORDER = ["authentic_learner", "matched_control", "artificial_learner"]
LABEL = {"authentic_learner": "authentic learner", "matched_control": "matched control",
         "artificial_learner": "artificial learner (AL)"}
COLOR = {"authentic_learner": "#d1495b", "matched_control": "#8d99ae", "artificial_learner": "#2a9d8f"}
ACQ_LABEL = {"sva": "SVA", "verb_morphology": "VERB:FORM", "tense": "TENSE",
             "determiner": "DET", "preposition": "PREP", "noun_number": "NOUN:NUM"}


def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.png / {name}.svg")


def _by_role(signals):
    """id-keyed sources -> role-keyed, in canonical ORDER (only roles present)."""
    out = {}
    for s in signals["sources"].values():
        out[s["role"]] = s
    return out


# ---------------------------------------------------------------------------
def fig1_length(signals, figdir):
    roles = _by_role(signals)
    present = [r for r in ORDER if r in roles]
    data = []
    for r in present:
        vals = [it["length_ratio"] for it in roles[r]["length_per_item"] if it.get("length_ratio")]
        data.append(vals or [1.0])
    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot(data, positions=range(len(present)), widths=0.5, patch_artist=True, showfliers=False)
    for patch, r in zip(bp["boxes"], present):
        patch.set_facecolor(COLOR[r]); patch.set_alpha(0.55)
    for med in bp["medians"]:
        med.set_color("#222")
    rng = np.random.default_rng(0)
    for i, (r, vals) in enumerate(zip(present, data)):
        xs = i + (rng.random(len(vals)) - 0.5) * 0.28
        ax.scatter(xs, vals, s=16, color=COLOR[r], edgecolor="#333", linewidth=0.3, zorder=3)
    ax.axhline(1.0, ls="--", color="#d1495b", lw=1.3, label="authentic-learner length (ratio = 1)")
    ax.set_yscale("log")
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([LABEL[r] for r in present], rotation=10, ha="right", fontsize=9)
    ax.set_ylabel("length ratio  (gen tokens / ref tokens, log)")
    ax.set_title("Fig 1 — Length over-generation (per continuation vs the authentic-learner reference)")
    ax.legend(fontsize=8, loc="upper left")
    _save(fig, figdir, "fig1_length_overgeneration")


def fig2_error_distance(signals, figdir):
    roles = _by_role(signals)
    present = [r for r in ORDER if r in roles]
    # per-source top tags (union of top-8 by count across sources)
    all_counts = {}
    for r in present:
        for t, c in roles[r]["gen_tag_counts"].items():
            all_counts[t] = all_counts.get(t, 0) + c
    tags = sorted(all_counts, key=lambda t: -all_counts[t])[:10]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(tags)); w = 0.8 / max(len(present), 1)
    for i, r in enumerate(present):
        jsd = roles[r]["jsd_to_learner"]
        lbl = f"{LABEL[r]} (JSD={jsd:.3f})"
        ax1.bar(x + i * w, [roles[r]["gen_tag_counts"].get(t, 0) for t in tags], w, label=lbl, color=COLOR[r])
    ax1.set_xticks(x + w * (len(present) - 1) / 2)
    ax1.set_xticklabels(tags, rotation=45, ha="right", fontsize=7)
    ax1.set_ylabel("generation-region count")
    ax1.set_title("Fig 2a — Error-tag distribution (JSD to authentic learner annotated)")
    ax1.legend(fontsize=7)
    # acquisition-category panel
    cats = list(ACQ_LABEL.keys())
    xa = np.arange(len(cats))
    for i, r in enumerate(present):
        ax2.bar(xa + i * w, [roles[r]["acquisition_counts"].get(c, 0) for c in cats], w, label=LABEL[r], color=COLOR[r])
    ax2.set_xticks(xa + w * (len(present) - 1) / 2)
    ax2.set_xticklabels([ACQ_LABEL[c] for c in cats], rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("count")
    ax2.set_title("Fig 2b — Acquisition categories")
    ax2.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, figdir, "fig2_error_tag_distance")


def fig3_distance_plane(plane, figdir):
    """THE money figure — 2-D distance-to-learner plane with the control→AL arrow."""
    pts = plane["points"]
    fig, ax = plt.subplots(figsize=(6, 5.4))
    ax.axhline(0, color="#d1495b", lw=1.0, ls="--", alpha=0.7)
    ax.axvline(0, color="#d1495b", lw=1.0, ls="--", alpha=0.7)
    for role in ORDER:
        if role not in pts:
            continue
        p = pts[role]
        xx = p["length_distance"] if p["length_distance"] is not None else 0.0
        yy = p["jsd"] if p["jsd"] is not None else 0.0
        ax.scatter([xx], [yy], s=170, color=COLOR[role], edgecolor="#222", zorder=4,
                   marker=("*" if role == "authentic_learner" else "o"),
                   label=f"{LABEL[role]}" + (" (yardstick, origin)" if role == "authentic_learner" else ""))
        ax.annotate(LABEL[role], (xx, yy), textcoords="offset points", xytext=(8, 6), fontsize=8)
    for arr in plane.get("arrows", []):
        c = pts["matched_control"]; a = pts["artificial_learner"]
        ax.annotate("", xy=(a["length_distance"], a["jsd"]),
                    xytext=(c["length_distance"], c["jsd"]),
                    arrowprops=dict(arrowstyle="-|>", color="#2a9d8f", lw=2.2, shrinkA=8, shrinkB=8))
        mx = (c["length_distance"] + a["length_distance"]) / 2
        my = (c["jsd"] + a["jsd"]) / 2
        tag = "control → AL" + ("  (inward on both ✓)" if arr.get("inward_both") else "")
        ax.annotate(tag, (mx, my), textcoords="offset points", xytext=(6, -12), fontsize=8, color="#1a6b60")
    ax.set_xlabel("length distance   |log(gen_len / ref_len)|   →  farther from learner")
    ax.set_ylabel("JSD to authentic learner   →  farther")
    ax.set_title("Fig 3 — Distance-to-learner plane (the centerpiece)\nauthentic learner at origin; arrow = matched pair control→AL")
    ax.legend(fontsize=8, loc="upper right")
    ax.margins(0.2)
    _save(fig, figdir, "fig3_distance_plane")


def fig4_movement(signals, figdir):
    roles = _by_role(signals)
    p = signals["pairs"][0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    # (a) length_ratio control vs AL vs learner
    present = [r for r in ORDER if r in roles]
    lrs = [(roles[r]["length"]["median_length_ratio"] or (1.0 if r == "authentic_learner" else None)) for r in present]
    ax1.bar(range(len(present)), [v or 0 for v in lrs], color=[COLOR[r] for r in present])
    ax1.axhline(1.0, ls="--", color="#d1495b", lw=1.2)
    ax1.set_xticks(range(len(present))); ax1.set_xticklabels([LABEL[r] for r in present], rotation=12, ha="right", fontsize=8)
    ax1.set_ylabel("median length ratio")
    ax1.set_title("Fig 4a — Length toward the yardstick (=1)")
    for i, v in enumerate(lrs):
        if v is not None:
            ax1.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    # (b) JSD control vs AL with RDR
    jc, ja = p["jsd_control"], p["jsd_al"]
    ax2.bar([0, 1], [jc, ja], color=[COLOR["matched_control"], COLOR["artificial_learner"]])
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["matched control", "artificial learner (AL)"], fontsize=8)
    ax2.set_ylabel("JSD to authentic learner")
    rdr = p["rdr"]
    ax2.set_title(f"Fig 4b — Error-tag distance (RDR = {rdr:.2f})" if rdr is not None else "Fig 4b — Error-tag distance")
    for i, v in enumerate([jc, ja]):
        ax2.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    _save(fig, figdir, "fig4_movement")


# ---------------------------------------------------------------------------
def _fmt(v, nd=3):
    if v is None:
        return "—"
    if isinstance(v, float):
        if v == float("inf"):
            return "∞"
        return f"{v:.{nd}f}"
    return str(v)


def table1_length(signals, tabdir):
    roles = _by_role(signals)
    present = [r for r in ORDER if r in roles]
    rows = [
        ("median length_ratio", lambda s: _fmt(s["length"]["median_length_ratio"], 2)),
        ("length_ratio IQR", lambda s: (f"[{_fmt(s['length']['length_ratio_iqr'][0],2)}, {_fmt(s['length']['length_ratio_iqr'][1],2)}]" if s["length"]["length_ratio_iqr"] else "—")),
        ("over_generation_rate (θ=1.5)", lambda s: _fmt(s["length"]["over_generation_rate"], 3)),
        ("mean n_sentences", lambda s: _fmt(s["length"]["mean_n_sentences"], 2)),
        ("multi-sentence rate", lambda s: _fmt(s["length"]["multi_sentence_rate"], 3)),
        ("ends-with-terminator rate (Tier-1 proxy for stop; true stop_reason=Tier-2)", lambda s: _fmt(s["length"]["ends_with_terminator_rate"], 3)),
        ("mean length_distance", lambda s: _fmt(s["length"]["mean_length_distance"], 3)),
    ]
    cols = present + (["Δ (AL − control)"] if ("artificial_learner" in roles and "matched_control" in roles) else [])
    header = "| metric | " + " | ".join(LABEL.get(c, c) for c in cols) + " |"
    sep = "|" + "|".join(["---"] * (len(cols) + 1)) + "|"
    lines = ["# Table 1 — Length metrics (this run)", "", header, sep]
    for name, fn in rows:
        cells = [fn(roles[r]) for r in present]
        if "Δ (AL − control)" in cols:
            va = roles["artificial_learner"]["length"]
            vc = roles["matched_control"]["length"]
            key = {"median length_ratio": "median_length_ratio", "over_generation_rate (θ=1.5)": "over_generation_rate",
                   "mean n_sentences": "mean_n_sentences", "mean length_distance": "mean_length_distance",
                   "multi-sentence rate": "multi_sentence_rate"}.get(name)
            if key and va.get(key) is not None and vc.get(key) is not None:
                cells.append(_fmt(va[key] - vc[key], 3))
            else:
                cells.append("—")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    _write(tabdir, "table1_length", "\n".join(lines))


def table2_matched_pairs(signals, tabdir):
    lines = ["# Table 2 — Matched-pair results", "",
             "| pair | len_ratio(AL) | len_ratio(ctrl) | Δlen-dist | JSD(AL) | JSD(ctrl) | RDR | SVA(AL/ctrl) | VERB:FORM(AL/ctrl) | DET(AL/ctrl) | err/sent(AL,ctrl) | PPL(AL,ctrl) | joint inward |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in signals["pairs"]:
        ar = p["acquisition_ratios"]
        lines.append(
            f"| {p['label']} | {_fmt(p['length_ratio_al'],2)} | {_fmt(p['length_ratio_control'],2)} | "
            f"{_fmt(p['delta_length_distance'],3)} | {_fmt(p['jsd_al'],3)} | {_fmt(p['jsd_control'],3)} | {_fmt(p['rdr'],3)} | "
            f"{_fmt(ar['sva']['al_over_control'],2)} | {_fmt(ar['verb_morphology']['al_over_control'],2)} | {_fmt(ar['determiner']['al_over_control'],2)} | "
            f"{_fmt(p['errors_per_sentence_al'],2)},{_fmt(p['errors_per_sentence_control'],2)} | "
            f"{_fmt(p['ppl_al'],1)},{_fmt(p['ppl_control'],1)} | {'yes' if p['joint_inward'] else 'no'} |")
    _write(tabdir, "table2_matched_pairs", "\n".join(lines))


def _write(tabdir, name, md):
    os.makedirs(tabdir, exist_ok=True)
    with open(os.path.join(tabdir, f"{name}.md"), "w") as f:
        f.write(md + "\n")
    print(f"  wrote {name}.md")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Two-signal figures + tables")
    ap.add_argument("--in", dest="indir", required=True, help="two_signal.py output dir (signals.json, distance_plane.json)")
    ap.add_argument("--out", dest="outdir", default=None, help="default: <in>")
    args = ap.parse_args(argv)
    indir = args.indir
    outdir = args.outdir or indir
    with open(os.path.join(indir, "signals.json")) as f:
        signals = json.load(f)
    with open(os.path.join(indir, "distance_plane.json")) as f:
        plane = json.load(f)
    figdir = os.path.join(outdir, "figures")
    tabdir = os.path.join(outdir, "tables")
    print("[two_signal_figures] figures:")
    fig1_length(signals, figdir)
    fig2_error_distance(signals, figdir)
    fig3_distance_plane(plane, figdir)
    fig4_movement(signals, figdir)
    print("[two_signal_figures] tables:")
    table1_length(signals, tabdir)
    table2_matched_pairs(signals, tabdir)
    print("TWO_SIGNAL_FIGURES_OK")


if __name__ == "__main__":
    main()
