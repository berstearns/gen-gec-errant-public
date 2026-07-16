"""Build the two shared precomputations every analysis reads (H1 step 1).

  _shared/paired_index.json   the intersection (text_id,sentence_idx) set +
                              per-source coverage
  _shared/distributions.json  per source x granularity {counts, shares},
                              restricted to the paired index, generation region

Built ONCE per run so no analysis re-derives them and results cannot drift.
"""
from __future__ import annotations

import json
import os
from collections import Counter

from . import common


def build_paired_index(sources: dict, run_slug: str) -> dict:
    """Intersect (text_id,sentence_idx) across ALL sources' full-sentence rosters
    (from full_results, so zero-error sentences are still paired)."""
    key_sets = {sid: set(s.keys_ordered) for sid, s in sources.items()}
    inter = set.intersection(*key_sets.values()) if key_sets else set()
    # Deterministic order: sort by (int text_id, int sentence_idx) when numeric.
    def _sort_key(k):
        try:
            return (0, int(k[0]), int(k[1]))
        except ValueError:
            return (1, k[0], k[1])
    keys = sorted(inter, key=_sort_key)
    if not keys:
        raise SystemExit("BLOCKER: paired index is empty — no shared (text_id,sentence_idx).")
    coverage = {
        sid: {"total_sentences": len(s.keys_ordered), "in_paired": len(inter & set(s.keys_ordered))}
        for sid, s in sources.items()
    }
    return {
        "run_slug": run_slug,
        "n": len(keys),
        "keys": [list(k) for k in keys],
        "sources": list(sources.keys()),
        "roles": {sid: s.role for sid, s in sources.items()},
        "coverage": coverage,
    }


def _granularity_counts(gen_tag_counts: Counter) -> dict:
    """Fold a per-tag Counter into the four granularities."""
    op: Counter = Counter()
    pos: Counter = Counter()
    acq: Counter = Counter()
    for tag, c in gen_tag_counts.items():
        op[common.operation(tag)] += c
        pos[common.pos_family(tag)] += c
        if tag in common.ACQUISITION_TAGS:
            acq[tag] += c
    return {
        "operation": _pack(op),
        "pos": _pack(pos),
        "tag": _pack(gen_tag_counts),
        "acq": _pack(acq),
    }


def _pack(counter: Counter) -> dict:
    counts = {k: int(v) for k, v in counter.items()}
    return {"counts": counts, "shares": common.shares(counts), "total": int(sum(counts.values()))}


def build_distributions(sources: dict, paired_keys: list) -> dict:
    keys = [tuple(k) for k in paired_keys]
    out = {"sources": {}, "roles": {sid: s.role for sid, s in sources.items()}}
    # global pos_map over the union tag vocabulary, for auditability (B2)
    pos_map = {}
    for sid, s in sources.items():
        tag_counts = s.gen_tag_counts(keys)
        for tag in tag_counts:
            pos_map[tag] = common.pos_family(tag)
        out["sources"][sid] = _granularity_counts(tag_counts)
    out["pos_map"] = dict(sorted(pos_map.items()))
    out["paired_n"] = len(keys)
    return out


def build_shared(sources: dict, run_slug: str, out_root: str) -> tuple[dict, dict]:
    shared_dir = os.path.join(out_root, "_shared")
    os.makedirs(shared_dir, exist_ok=True)
    paired = build_paired_index(sources, run_slug)
    with open(os.path.join(shared_dir, "paired_index.json"), "w") as f:
        json.dump(paired, f, indent=2)
        f.write("\n")
    dists = build_distributions(sources, paired["keys"])
    with open(os.path.join(shared_dir, "distributions.json"), "w") as f:
        json.dump(dists, f, indent=2)
        f.write("\n")
    return paired, dists
