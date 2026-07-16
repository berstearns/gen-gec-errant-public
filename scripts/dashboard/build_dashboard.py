#!/usr/bin/env python3
"""Build a self-contained HTML dashboard for the error-dynamics analysis.

Reads existing pipeline artifacts (raw_results.json + *_summary.json) and, when
present, the analysis harness outputs under analysis-outputs/<slug>/*/result.json
(paired + CI versions — preferred). Emits ONE standalone HTML file (inline CSS +
inline SVG charts, no external deps, theme-aware) so an analyst can just open it.

Exploratory: tiny-sample values are hypothesis-generating, not confirmatory.
"""
import json, math, os, sys, html, datetime

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SLUG = "tiny-sample-2026-07-06"
OUT = os.path.join(REPO, "analysis-outputs", "dashboard.html")

SOURCES = {  # role -> (raw_results path, key inside)
    "LEARNER":  ("outputs/s1-pilot/gpt2-small/raw_results.json", "learner_baseline"),
    "AL":       ("outputs/s1-pilot/ft-gpt2-small/raw_results.json", "ft-gpt2-small"),
    "CONTROL":  ("outputs/s1-pilot/gpt2-small/raw_results.json", "gpt2-small"),
}
SUMMARY = {
    "LEARNER": ("outputs/s1-pilot/gpt2-small/learner_baseline_summary.json"),
    "AL":      ("outputs/s1-pilot/ft-gpt2-small/ft-gpt2-small_summary.json"),
    "CONTROL": ("outputs/s1-pilot/gpt2-small/gpt2-small_summary.json"),
}
ACQ = ["R:VERB:SVA", "M:VERB:FORM", "R:VERB:FORM", "R:VERB:TENSE",
       "M:DET", "U:DET", "R:DET", "M:PREP", "R:PREP", "U:PREP", "R:NOUN:NUM"]
ARTIFACT = {"R:ORTH", "R:SPELL", "M:PUNCT", "R:PUNCT", "U:PUNCT"}
COL = {"LEARNER": "#4c9f70", "AL": "#e08a3c", "CONTROL": "#7a86c9"}


def load(path):
    p = os.path.join(REPO, path)
    return json.load(open(p)) if os.path.exists(p) else None


def gen_counts(role):
    path, key = SOURCES[role]
    d = load(path)
    if not d or key not in d:
        return {}
    return dict(d[key].get("region_error_summary", {}).get("generation_error_type_counts", {}))


def norm(c):
    t = sum(c.values()) or 1
    return {k: v / t for k, v in c.items()}


def jsd(p, q):
    ks = set(p) | set(q)
    p = {k: p.get(k, 0) for k in ks}; q = {k: q.get(k, 0) for k in ks}
    m = {k: (p[k] + q[k]) / 2 for k in ks}
    kl = lambda a, b: sum(a[k] * math.log2(a[k] / b[k]) for k in ks if a[k] > 0)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def agg(counts, mode):
    """Re-aggregate a tag->count dict to operation or pos granularity."""
    out = {}
    for tag, c in counts.items():
        parts = tag.split(":")
        if mode == "operation":
            k = parts[0]
        elif mode == "pos":
            k = parts[1] if len(parts) > 1 else parts[0]
        else:
            k = tag
        out[k] = out.get(k, 0) + c
    return out


# ---------- SVG helpers ----------
def esc(s):
    return html.escape(str(s))


def bar_grouped(title, cats, series, maxv=None, unit="%"):
    """series: dict role->{cat:val}. Horizontal grouped bars."""
    roles = [r for r in ("LEARNER", "AL", "CONTROL") if r in series]
    maxv = maxv or max([series[r].get(c, 0) for r in roles for c in cats] + [1e-9])
    rowh, gap, bw = 18, 26, 460
    H = len(cats) * (len(roles) * rowh + gap) + 40
    y = 24
    rows = []
    for c in cats:
        rows.append(f'<text x="0" y="{y-6}" class="cl">{esc(c)}</text>')
        for r in roles:
            v = series[r].get(c, 0)
            w = max(1, v / maxv * bw)
            lbl = f"{v*100:.1f}%" if unit == "%" else f"{v:.2f}"
            rows.append(
                f'<rect x="150" y="{y}" width="{w:.1f}" height="{rowh-4}" fill="{COL[r]}" rx="2"/>'
                f'<text x="{150+w+5:.1f}" y="{y+rowh-8}" class="vl">{lbl}</text>')
            y += rowh
        y += gap - rowh
    leg = " ".join(
        f'<span class="lg"><i style="background:{COL[r]}"></i>{r}</span>' for r in roles)
    return (f'<div class="chart"><h4>{esc(title)}</h4>'
            f'<div class="legend">{leg}</div>'
            f'<svg viewBox="0 0 640 {H}" width="100%">{"".join(rows)}</svg></div>')


def table(headers, rows, cls=""):
    h = "".join(f"<th>{esc(x)}</th>" for x in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>'


def card(label, value, sub="", good=None):
    cls = "card" + ("" if good is None else (" good" if good else " bad"))
    return f'<div class="{cls}"><div class="cv">{esc(value)}</div><div class="cl2">{esc(label)}</div><div class="cs">{esc(sub)}</div></div>'


# ---------- build ----------
def main():
    counts = {r: gen_counts(r) for r in SOURCES}
    present = [r for r in ("LEARNER", "AL", "CONTROL") if counts.get(r)]
    n = {}
    for r in present:
        s = load(SUMMARY[r]) or {}
        es = s.get("error_summary", s)
        n[r] = {"total_gen_err": sum(counts[r].values()),
                "err_per_sent": es.get("avg_errors_per_sentence"),
                "sents": es.get("total_sentences")}

    dist = {g: {r: norm(agg(counts[r], g)) for r in present}
            for g in ("operation", "pos", "tag")}

    jsd_tbl = {}
    for g in ("operation", "pos", "tag"):
        row = {}
        if "LEARNER" in dist[g]:
            for r in ("AL", "CONTROL"):
                if r in dist[g]:
                    row[r] = jsd(dist[g][r], dist[g]["LEARNER"])
        jsd_tbl[g] = row

    al_closer = (jsd_tbl["tag"].get("AL", 9) < jsd_tbl["tag"].get("CONTROL", 9))

    # top tags (by learner share)
    ltags = sorted(dist["tag"].get("LEARNER", {}), key=lambda k: -dist["tag"]["LEARNER"][k])[:10]
    # acquisition present
    acq_present = [t for t in ACQ if any(t in dist["tag"][r] for r in present)]
    # artifact share
    art = {r: sum(v for k, v in dist["tag"][r].items() if k in ARTIFACT) for r in present}

    # sections
    S = []

    # ---- Z1 findings lead the dashboard (the conclusions, not the charts) ----
    zpath = os.path.join(REPO, "analysis-outputs", SLUG, "Z1-findings-synthesis", "findings.json")
    if os.path.exists(zpath):
        try:
            zf = json.load(open(zpath))
            bp = zf.get("by_pair", {})
            pair = next(iter(bp)) if bp else None
            finds = (bp[pair].get("findings") if isinstance(bp.get(pair), dict) else bp.get(pair)) if pair else []
        except Exception:
            finds = []
        if finds:
            S.append('<section class="findings"><h2>Findings — what the data concludes</h2>'
                     f'<p class="note">Synthesised by Z1 (pair {esc(pair)}). Claim → effect size → mechanism → confidence. EXPLORATORY (n=100): directional, significance deferred to S1.</p>')
            dirmap = {"toward_learner": ("→ learner", "good"), "away": ("✗ away", "bad"),
                      "mixed": ("~ mixed", ""), "none": ("· none", "")}
            for fd in finds:
                dl, dc = dirmap.get(fd.get("direction", ""), (fd.get("direction", ""), ""))
                es = fd.get("effect_size", {})
                interp = es.get("interp", "") if isinstance(es, dict) else ""
                conf = fd.get("confidence", {})
                conf = conf.get("level", "") if isinstance(conf, dict) else conf
                S.append(
                    f'<div class="finding {dc}"><div class="fh"><span class="fid">{esc(fd.get("id",""))}</span>'
                    f'<span class="ftitle">{esc(fd.get("title",""))}</span>'
                    f'<span class="fdir {dc}">{esc(dl)}</span></div>'
                    f'<div class="fclaim">{esc(fd.get("claim",""))}</div>'
                    f'<div class="fmeta"><b>effect:</b> {esc(interp)}</div>'
                    f'<div class="fmeta"><b>mechanism:</b> {esc(fd.get("mechanism",""))}</div>'
                    f'<div class="fmeta conf"><b>confidence:</b> {esc(conf)}</div></div>')
            S.append('</section>')

    S.append('<section><h2>α · Distributional similarity</h2>')
    cards = []
    if "AL" in jsd_tbl["tag"]:
        cards.append(card("JSD  AL → LEARNER", f'{jsd_tbl["tag"]["AL"]:.3f}', "tag-level (↓ closer)", good=al_closer))
    if "CONTROL" in jsd_tbl["tag"]:
        cards.append(card("JSD  CONTROL → LEARNER", f'{jsd_tbl["tag"]["CONTROL"]:.3f}', "tag-level baseline"))
    cards.append(card("AL closer than control?", "YES ✓" if al_closer else "NO ✗",
                      "direction of the C1 hypothesis", good=al_closer))
    S.append('<div class="cards">' + "".join(cards) + '</div>')
    S.append(bar_grouped("JSD to real learners, by granularity (lower = closer)",
                         ["operation", "pos", "tag"],
                         {"AL": {g: jsd_tbl[g].get("AL", 0) for g in ("operation", "pos", "tag")},
                          "CONTROL": {g: jsd_tbl[g].get("CONTROL", 0) for g in ("operation", "pos", "tag")}},
                         unit="x"))
    S.append('</section>')

    S.append('<section><h2>β · Where the errors sit</h2>')
    S.append(bar_grouped("Top-10 learner error tags — share of each source's gen-region errors",
                         ltags, {r: dist["tag"][r] for r in present}))
    S.append(bar_grouped("Operation balance (Missing / Replacement / Unnecessary)",
                         ["M", "R", "U"], {r: dist["operation"][r] for r in present}))
    if acq_present:
        S.append(bar_grouped("Acquisition-relevant categories (SLA-diagnostic)",
                             acq_present, {r: dist["tag"][r] for r in present}))
    S.append('</section>')

    S.append('<section><h2>γ · Magnitude</h2>')
    mag_rows = [[r, f'{n[r]["err_per_sent"]:.2f}' if n[r]["err_per_sent"] else "—",
                 n[r]["total_gen_err"], n[r]["sents"] or "—"] for r in present]
    S.append(table(["source", "errors / sentence", "gen-region errors", "sentences"], mag_rows, "wide"))
    S.append('</section>')

    S.append('<section><h2>δ · Validity — GEC-artifact share</h2>')
    S.append('<p class="note">Share of each source\'s gen-region errors that are orthography/spelling/punctuation '
             '(R:ORTH, R:SPELL, *:PUNCT) — possible corrector artifacts, not interlanguage. '
             'A large AL excess here would mean the similarity is partly artifactual (see spec D4).</p>')
    S.append(bar_grouped("Artifact-class share of errors", ["artifact"],
                         {r: {"artifact": art[r]} for r in present}))
    S.append('</section>')

    # harness status
    harness = os.path.join(REPO, "analysis-outputs", SLUG)
    ran = []
    if os.path.isdir(harness):
        for d in sorted(os.listdir(harness)):
            rj = os.path.join(harness, d, "result.json")
            if os.path.isfile(rj):
                try:
                    st = json.load(open(rj)).get("status", "?")
                except Exception:
                    st = "?"
                ran.append((d, st))
    hrows = [[d, ("✅ done" if st not in ("not_run", "skipped", "?") else f"⏳ {st}")] for d, st in ran]
    S.append('<section><h2>Analysis harness (dispatch #6)</h2>')
    S.append('<p class="note">This dashboard computes the headline views directly from pipeline artifacts. '
             'When the full harness (paired shared-ID restriction + bootstrap CIs + D1–D4 screens, per '
             'gen-gec-review-specs/analysis/) populates the tree below, re-open to see the rigorous versions.</p>')
    S.append(table(["analysis", "status"], hrows or [["(none yet)", "—"]]))
    S.append('</section>')

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    npar = " · ".join(f"{r} n={n[r]['sents']}" for r in present if n[r]['sents'])
    doc = TEMPLATE.format(ts=ts, npar=esc(npar), body="\n".join(S))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write(doc)
    print(f"[dashboard] wrote {OUT}  ({ts})  sources: {', '.join(present)}")


TEMPLATE = """<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="60">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AL vs Learner — Error Dynamics</title><style>
:root{{--bg:#fbfbfd;--fg:#1b1c1f;--mut:#6b7076;--card:#fff;--bd:#e3e5ea;--acc:#4c9f70}}
@media(prefers-color-scheme:dark){{:root{{--bg:#15171b;--fg:#e8eaed;--mut:#9aa0a8;--card:#1d2025;--bd:#2c3038}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}}
header{{padding:24px 28px;border-bottom:1px solid var(--bd)}}
h1{{margin:0 0 4px;font-size:20px}}.sub{{color:var(--mut);font-size:13px}}
.banner{{background:#e08a3c22;border:1px solid #e08a3c66;color:var(--fg);
padding:8px 14px;border-radius:8px;margin:14px 28px;font-size:13px}}
main{{max-width:900px;margin:0 auto;padding:8px 28px 60px}}
section{{margin:26px 0;padding:18px;background:var(--card);border:1px solid var(--bd);border-radius:12px}}
h2{{margin:0 0 14px;font-size:16px}}h4{{margin:6px 0;font-size:13px;color:var(--mut);font-weight:600}}
.cards{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px}}
.card{{flex:1;min-width:150px;padding:14px;border:1px solid var(--bd);border-radius:10px;background:var(--bg)}}
.card.good{{border-color:#4c9f70}}.card.bad{{border-color:#c65c5c}}
.cv{{font-size:26px;font-weight:700}}.cl2{{font-size:12px;margin-top:4px}}.cs{{font-size:11px;color:var(--mut)}}
.chart{{margin:16px 0;overflow-x:auto}}.legend{{font-size:11px;color:var(--mut);margin:2px 0 6px}}
.lg{{margin-right:12px}}.lg i{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:middle}}
text.cl{{font-size:11px;fill:var(--fg);font-weight:600}}text.vl{{font-size:10px;fill:var(--mut)}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}}
th,td{{text-align:left;padding:6px 10px;border-bottom:1px solid var(--bd)}}th{{color:var(--mut);font-weight:600}}
.note{{font-size:12px;color:var(--mut);margin:4px 0 10px}}
section.findings{{border-color:var(--acc)}}
.finding{{border:1px solid var(--bd);border-left:4px solid var(--mut);border-radius:8px;padding:12px 14px;margin:10px 0;background:var(--bg)}}
.finding.good{{border-left-color:#4c9f70}}.finding.bad{{border-left-color:#c65c5c}}
.fh{{display:flex;align-items:center;gap:8px;margin-bottom:4px}}
.fid{{font-size:11px;font-weight:700;color:var(--mut);border:1px solid var(--bd);border-radius:4px;padding:1px 6px}}
.ftitle{{font-weight:700;font-size:14px;flex:1}}
.fdir{{font-size:11px;font-weight:600}}.fdir.good{{color:#4c9f70}}.fdir.bad{{color:#c65c5c}}
.fclaim{{font-size:13px;margin:4px 0}}
.fmeta{{font-size:12px;color:var(--mut);margin:2px 0}}.fmeta.conf{{font-style:italic}}
footer{{color:var(--mut);font-size:12px;padding:20px 28px;border-top:1px solid var(--bd)}}
</style></head><body>
<header><h1>Authentic vs Artificial Learner — Error Dynamics</h1>
<div class="sub">tiny-sample-2026-07-06 · {npar} · auto-refresh 60s · built {ts}</div></header>
<div class="banner"><b>EXPLORATORY</b> — tiny-sample, hypothesis-generating only. Not confirmatory
(single pair, unpaired n; the pre-registered verdict comes from full S1). Values computed directly
from pipeline artifacts; the rigorous paired/CI harness (dispatch #6) supersedes them when ready.</div>
<main>{body}</main>
<footer>Specs: gen-gec-review-specs/analysis/ · Instrument: coedit-large + ERRANT, sentence-wise,
generation-region errors · gen-gec-errant reviewer↔executor loop</footer></body></html>"""


if __name__ == "__main__":
    main()
