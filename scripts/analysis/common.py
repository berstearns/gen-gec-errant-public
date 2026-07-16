"""Shared helpers for the error-dynamics analysis harness.

Everything that more than one analysis module needs lives here: canonical
loaders for the pipeline artifacts, the distance-metric family, the ERRANT
tag taxonomy (POS mapping, acquisition set, artifact set), and the
OUTPUT-CONTRACT writers. Analysis modules stay pure w.r.t. their declared
inputs so results are deterministic and scale-invariant.

Spec: gen-gec-review-specs/analysis/{RESOURCES,OUTPUT-CONTRACT,H1-*}.md
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# ERRANT taxonomy
# ---------------------------------------------------------------------------

# Acquisition-critical tags (B4 / 20-preregistration.md). Grouped by phenomenon.
ACQUISITION_PHENOMENA: dict[str, list[str]] = {
    "sva": ["R:VERB:SVA"],
    "verb_morphology": ["M:VERB:FORM", "R:VERB:FORM"],
    "tense": ["R:VERB:TENSE"],
    "determiner": ["U:DET", "M:DET", "R:DET"],
    "preposition": ["M:PREP", "R:PREP", "U:PREP"],
    "noun_number": ["R:NOUN:NUM", "R:NOUN:INFL"],
}
ACQUISITION_TAGS: list[str] = [t for tags in ACQUISITION_PHENOMENA.values() for t in tags]

# GEC-instrument artifact classes (D4).
ARTIFACT_TAGS: set[str] = {"R:ORTH", "R:SPELL", "M:PUNCT", "R:PUNCT", "U:PUNCT"}


def pos_family(error_type: str) -> str:
    """POS head of an ERRANT tag: the token after the operation, before any
    finer subcategory. e.g. R:VERB:SVA->VERB, U:PUNCT->PUNCT, R:ORTH->ORTH,
    M:NOUN:NUM->NOUN. Operation-only edge cases fall back to OTHER."""
    parts = error_type.split(":")
    if len(parts) < 2:
        return "OTHER"
    return parts[1]


def operation(error_type: str) -> str:
    """M | R | U from the tag prefix."""
    return error_type.split(":", 1)[0]


# ---------------------------------------------------------------------------
# Distance-metric family (A1). All base-2, over an explicit union vocabulary.
# ---------------------------------------------------------------------------

def _aligned(counts_a: dict[str, float], counts_b: dict[str, float]):
    """Return (vocab, p, q) as normalized share vectors over the union vocab."""
    vocab = sorted(set(counts_a) | set(counts_b))
    a = np.array([counts_a.get(t, 0.0) for t in vocab], dtype=float)
    b = np.array([counts_b.get(t, 0.0) for t in vocab], dtype=float)
    p = a / a.sum() if a.sum() > 0 else np.zeros_like(a)
    q = b / b.sum() if b.sum() > 0 else np.zeros_like(b)
    return vocab, p, q


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence, base 2, symmetric, bounded [0,1]."""
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def kl_smoothed(p: np.ndarray, q: np.ndarray, eps: float) -> float:
    """KL(p || q), base 2, additive-eps smoothed + renormalized so it is finite."""
    ps = (p + eps)
    ps = ps / ps.sum()
    qs = (q + eps)
    qs = qs / qs.sum()
    return float(np.sum(ps * np.log2(ps / qs)))


def tvd(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation distance = 0.5 * sum|p-q|, bounded [0,1]."""
    return float(0.5 * np.sum(np.abs(p - q)))


def cross_entropy(p: np.ndarray, q: np.ndarray, eps: float) -> float:
    """H(p, q) = -sum p*log2(q), q eps-smoothed. p=learner, q=source."""
    qs = (q + eps)
    qs = qs / qs.sum()
    mask = p > 0
    return float(-np.sum(p[mask] * np.log2(qs[mask])))


def jsd_counts(a_counts: dict, b_counts: dict) -> float:
    """JSD between two raw-count dicts, aligned on their own pairwise union vocab.
    Use this instead of mixing `p`/`q` from separate _aligned() calls."""
    _, p, q = _aligned(a_counts, b_counts)
    return jsd(p, q)


def all_metrics(learner_counts: dict, source_counts: dict, eps: float) -> dict:
    """The full A1 metric block for one (source vs learner) pair."""
    _, p_l, q_s = _aligned(learner_counts, source_counts)
    return {
        "jsd": jsd(p_l, q_s),
        "kl_sl": kl_smoothed(q_s, p_l, eps),   # KL(source || learner)
        "kl_ls": kl_smoothed(p_l, q_s, eps),   # KL(learner || source)
        "tvd": tvd(p_l, q_s),
        "xent": cross_entropy(p_l, q_s, eps),  # H(learner, source)
        "n_learner": int(sum(learner_counts.values())),
        "n_source": int(sum(source_counts.values())),
    }


def shares(counts: dict[str, float]) -> dict[str, float]:
    tot = sum(counts.values())
    if tot == 0:
        return {k: 0.0 for k in counts}
    return {k: v / tot for k, v in counts.items()}


# ---------------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------------

def _col_prefix(model_id: str) -> str:
    """full_results.tsv column prefix for a model id (hyphens -> underscores)."""
    return model_id.replace("-", "_")


@dataclass
class Source:
    id: str
    role: str            # LEARNER | AL | CONTROL
    run_dir: str
    # ordered (text_id, sentence_idx) keys as they appear in full_results.tsv
    keys_ordered: list[tuple[str, str]]
    # per-key generation-region error rows (source=='full_text', region=='generation')
    gen_errors: dict[tuple[str, str], list[dict]]
    # per-key prompt-region error rows (for D1)
    prompt_errors: dict[tuple[str, str], list[dict]]
    # per-key generated/reference continuation text
    continuations: dict[tuple[str, str], str]
    # per-key perplexity (own-model; not cross-family comparable)
    perplexities: dict[tuple[str, str], float]
    # per-key prompt/continuation char boundary in the full_text
    prompt_boundaries: dict[tuple[str, str], int] = field(default_factory=dict)

    def gen_tag_counts(self, keys: list[tuple[str, str]]) -> Counter:
        c: Counter = Counter()
        for k in keys:
            for row in self.gen_errors.get(k, []):
                c[row["error_type"]] += 1
        return c

    def per_sentence_tag_counters(self, keys: list[tuple[str, str]]) -> dict:
        out = {}
        for k in keys:
            c: Counter = Counter()
            for row in self.gen_errors.get(k, []):
                c[row["error_type"]] += 1
            out[k] = c
        return out


def _load_errors_long(run_dir: str) -> list[dict]:
    path = os.path.join(run_dir, "errors_long_format.tsv")
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rows.append(row)
    return rows


def _load_full_results_keys(run_dir: str) -> list[tuple[str, str]]:
    """Ordered (text_id, sentence_idx) roster — one row per generated sentence,
    including zero-error sentences (which never appear in errors_long)."""
    path = os.path.join(run_dir, "full_results.tsv")
    keys = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            keys.append((row["text_id"], row["sentence_idx"]))
    return keys


def load_prompts(run_dir: str) -> dict[tuple[str, str], str]:
    """(text_id,sentence_idx) -> shared prompt text, from full_results.tsv."""
    path = os.path.join(run_dir, "full_results.tsv")
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            out[(row["text_id"], row["sentence_idx"])] = row.get("prompt", "")
    return out


def load_source(source_id: str, role: str, run_dir: str) -> Source:
    """Assemble a Source from a run directory. `source_id` is the value in the
    errors_long `model` column and the raw_results top-level key."""
    keys_ordered = _load_full_results_keys(run_dir)

    gen_errors: dict = {}
    prompt_errors: dict = {}
    for row in _load_errors_long(run_dir):
        if row["model"] != source_id:
            continue
        if row["source"] != "full_text":
            continue  # 'continuation'-source rows are fragment-annotated, artifact-heavy
        k = (row["text_id"], row["sentence_idx"])
        if row["region"] == "generation":
            gen_errors.setdefault(k, []).append(row)
        elif row["region"] == "prompt":
            prompt_errors.setdefault(k, []).append(row)

    # continuations + perplexities from raw_results (ordered same as full_results)
    raw_path = os.path.join(run_dir, "raw_results.json")
    with open(raw_path) as f:
        raw = json.load(f)
    block = raw[source_id]
    conts = block.get("continuations", [])
    ppls = block.get("perplexities", [])
    bounds = block.get("prompt_boundaries", [])
    continuations = {}
    perplexities = {}
    prompt_boundaries = {}
    for i, k in enumerate(keys_ordered):
        if i < len(conts):
            continuations[k] = conts[i]
        if i < len(ppls):
            perplexities[k] = ppls[i]
        if i < len(bounds):
            prompt_boundaries[k] = bounds[i]

    return Source(
        id=source_id, role=role, run_dir=run_dir,
        keys_ordered=keys_ordered, gen_errors=gen_errors,
        prompt_errors=prompt_errors, continuations=continuations,
        perplexities=perplexities, prompt_boundaries=prompt_boundaries,
    )


# ---------------------------------------------------------------------------
# spaCy tokenizer (matches ERRANT parsing) for length / density
# ---------------------------------------------------------------------------

_NLP = None


def token_count(text: str) -> int:
    """Whitespace-free token count using the same tokenizer ERRANT parses with
    (en_core_web_sm), so density denominators match the error tokenization."""
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm", disable=["parser", "tagger", "ner", "lemmatizer"])
    if not text or not text.strip():
        return 0
    return len([t for t in _NLP.tokenizer(text) if not t.is_space])


# ---------------------------------------------------------------------------
# OUTPUT-CONTRACT writers
# ---------------------------------------------------------------------------

def _canonical(obj: Any) -> str:
    """Deterministic JSON serialization for hashing / --verify."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_result(outdir: str, analysis_id: str, run_slug: str, n: dict,
                 params: dict, results: dict, caveats: list[str]) -> dict:
    """Write result.json (the machine-readable contract). Returns the object."""
    obj = {
        "analysis_id": analysis_id,
        "run_slug": run_slug,
        "n": n,
        "params": params,
        "results": results,
        "caveats": caveats,
    }
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "result.json"), "w") as f:
        f.write(json.dumps(obj, indent=2, ensure_ascii=False))
        f.write("\n")
    return obj


def write_md(outdir: str, text: str) -> None:
    with open(os.path.join(outdir, "result.md"), "w") as f:
        f.write(text.rstrip() + "\n")


def write_md_named(outdir: str, name: str, text: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, name), "w") as f:
        f.write(text.rstrip() + "\n")


def write_inputs(outdir: str, run_slug: str, sources: dict, extra: dict) -> None:
    """inputs.json — local provenance echo of exact resources consulted."""
    obj = {
        "run_slug": run_slug,
        "sources": {sid: {"role": s.role, "run_dir": s.run_dir}
                    for sid, s in sources.items()},
    }
    obj.update(extra)
    with open(os.path.join(outdir, "inputs.json"), "w") as f:
        f.write(json.dumps(obj, indent=2, ensure_ascii=False))
        f.write("\n")


def figures_dir(outdir: str) -> str:
    d = os.path.join(outdir, "figures")
    os.makedirs(d, exist_ok=True)
    return d


def save_csv(path: str, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


# ---------------------------------------------------------------------------
# Run context + multi-pair helpers
# ---------------------------------------------------------------------------

@dataclass
class Context:
    run_slug: str
    sources: dict            # id -> Source
    learner_id: str
    pairs: list              # list[(al_id, control_id)]
    paired_keys: list        # list[tuple(text_id, sentence_idx)]
    paired: dict             # paired_index.json
    dists: dict              # distributions.json
    out_root: str
    params: dict = field(default_factory=dict)
    exploratory: bool = True

    @property
    def learner(self) -> "Source":
        return self.sources[self.learner_id]

    def pair_label(self, al_id: str, ctrl_id: str) -> str:
        return f"{al_id}:{ctrl_id}"


def finalize_pairs(pair_results: dict) -> dict:
    """Wrap per-pair blocks as {by_pair:{label:block}} and, for the canonical
    single-pair (tiny-sample) case, alias the first pair's AL/CONTROL keys to
    the top level so result.json reads exactly as the specs describe. Full S1
    (5 pairs) leaves everything under by_pair — same key shape, no code edit."""
    out = {"by_pair": pair_results}
    if len(pair_results) == 1:
        only = next(iter(pair_results.values()))
        for k, v in only.items():
            out[k] = v
    return out
