"""GPT-2 family × generation stopping-strategies × AL-vs-control.

The ONLY novel axis here is *generation stopping* (4 strategies). Everything downstream —
GEC, ERRANT annotation, the two signals (length + error-tag JSD), RDR, the six acquisition
categories — is reused from the batch pipeline building blocks and `scripts.analysis.two_signal`
(the batch ``src/`` is untouched). This module is importable and runnable standalone; the Colab
notebook is a thin runner over ``run_experiment``.

Stopping strategies (one generation config each; all share temp 1.0 / top_k 50 / top_p 0.95 /
rep_pen 1.2 / no_repeat_ngram_size 3):
  hard_cap        max_new=64, min_new=64  → forced 64 tokens, no early stop  (over-generation baseline)
  natural_eos     max_new=64, min_new=1   → stop on the model's own EOS       (does it choose to stop?)
  sentence_stop   natural + SentenceStop  → stop at the first . ! ?           (one-sentence completion)
Per generation we record ``stop_reason ∈ {eos, sentence, cap}`` + ``truncated``.

(``length_matched`` — the oracle length-ceiling that *truncated* hard_cap to the learner's own
token length — was DROPPED per REALIGN / DISPATCH #11: truncation (not generation) contaminates
the error comparison. Re-add later, if ever, as force-generate-N and excluded from any error
comparison.)
"""
from __future__ import annotations

import gc
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from gen_gec_errant.generation.config import GenerationParams, ModelConfig
from gen_gec_errant.generation.runner import get_device, load_model, compute_perplexity
from gen_gec_errant.gec.runner import load_gec_corrector, correct_in_batches
from gen_gec_errant.gec.config import GECConfig
from gen_gec_errant.annotation.runner import (
    run_annotation, classify_errors_by_region, summarize_errors_by_region,
)
from gen_gec_errant.annotation.config import AnnotationConfig
from gen_gec_errant.data_loader.runner import run_data_loader
from gen_gec_errant.data_loader.config import DataLoaderConfig
from gen_gec_errant.pipeline.runner import _serialize_annotations

STRATEGIES = ["hard_cap", "natural_eos", "sentence_stop"]
# GGE_STRATEGIES (comma-sep subset, e.g. "sentence_stop") restricts the run to those strategies —
# the unselected generate passes are SKIPPED entirely (an S1 full-corpus run of sentence_stop alone
# is ~10x cheaper than the 3-strategy matrix: the other two write ~6.3x the learner's length and
# GEC/ERRANT time scales with text volume). Unset/empty = all three. Unknown names fail loud.
# Read via layered_get (env > .env > Colab userdata > default) so every delivery channel works;
# fall back to the bare env off-tree (layered_get needs the installed package).
try:
    from gen_gec_errant.nbenv import layered_get as _layered_get
    _env_strategies = _layered_get("GGE_STRATEGIES", "").strip()
except ImportError:
    _env_strategies = os.environ.get("GGE_STRATEGIES", "").strip()
if _env_strategies:
    _requested = [x.strip() for x in _env_strategies.split(",") if x.strip()]
    _unknown = [x for x in _requested if x not in STRATEGIES]
    if _unknown:
        raise ValueError(f"GGE_STRATEGIES has unknown strategies {_unknown}; valid: {STRATEGIES}")
    STRATEGIES = [s for s in STRATEGIES if s in _requested]   # canonical order preserved
# length_matched (an oracle length-ceiling that TRUNCATED hard_cap to the learner's own token
# length, forcing length_ratio ≈ 1.0 by construction) was DROPPED per REALIGN / DISPATCH #11:
# truncation contaminates the error comparison. There is no oracle strategy now — every strategy
# is a genuine, comparable stopping rule.
ORACLE_STRATEGIES: tuple = ()
COMPARABLE_STRATEGIES = [s for s in STRATEGIES if s not in ORACLE_STRATEGIES]
JSD_CEIL = 1.0        # Jensen–Shannon divergence (log base 2) upper bound
JSD_MIN_N = 20        # below this many scored sentences the error-tag JSD is unreliable (sparse support)
SENTENCE_ENDERS = ".!?"
MAX_NEW = 64
BASE_DECODE = dict(temperature=1.0, top_k=50, top_p=0.95, do_sample=True, repetition_penalty=1.2)


# ── the notebook-local sentence StoppingCriteria ────────────────────────────

class SentenceStopCriteria(StoppingCriteria):
    """Stop each sequence once it has generated a sentence terminator (``. ! ?``) in
    the CONTINUATION (past ``prompt_len``). Returns a per-row bool (transformers >=4.4x
    honours per-sequence stopping via ``all()`` over the batch, so we also expose
    ``done`` for post-hoc stop_reason)."""

    def __init__(self, tokenizer, prompt_lens: torch.Tensor):
        self.tok = tokenizer
        self.prompt_lens = prompt_lens
        # Must live on the SAME device as the generation: transformers combines this criteria's
        # return with `unfinished_sequences` (on the model's device), so a CPU tensor raises a
        # device mismatch the moment the matrix runs on the T4. prompt_lens is derived from the
        # already-moved `inputs`, so it is the model's device by construction.
        self.done = torch.zeros(len(prompt_lens), dtype=torch.bool, device=prompt_lens.device)

    def __call__(self, input_ids: torch.LongTensor, scores, **kwargs) -> torch.BoolTensor:
        for i in range(input_ids.shape[0]):
            if self.done[i]:
                continue
            gen_ids = input_ids[i, int(self.prompt_lens[i]):]
            text = self.tok.decode(gen_ids, skip_special_tokens=True)
            if any(ch in text for ch in SENTENCE_ENDERS):
                self.done[i] = True
        return self.done.clone()


def _truncate_at_sentence(text: str) -> str:
    """Keep text up to and including the first . ! ? (post-hoc twin of SentenceStop)."""
    idx = min([text.find(c) for c in SENTENCE_ENDERS if text.find(c) >= 0] or [-1])
    return text[: idx + 1] if idx >= 0 else text


# ── generation: all 4 strategies from 2 model.generate passes ───────────────

@torch.no_grad()
def _gen(model, tok, inputs, prompt_lens, *, min_new, max_new, criteria=None):
    kw = dict(BASE_DECODE, no_repeat_ngram_size=3, max_new_tokens=max_new, min_new_tokens=min_new,
              pad_token_id=tok.pad_token_id)
    if criteria is not None:
        kw["stopping_criteria"] = StoppingCriteriaList([criteria])
    out = model.generate(**inputs, **kw)
    conts, newlens = [], []
    for j, ids in enumerate(out):
        gen_ids = ids[int(prompt_lens[j]):]
        newlens.append(int((gen_ids != tok.pad_token_id).sum()))
        conts.append(tok.decode(gen_ids, skip_special_tokens=True).strip())
    return conts, newlens


@torch.no_grad()
def generate_all_strategies(model, tok, prompts: List[str], references: List[str],
                            device, batch_size: int = 8) -> Dict[str, dict]:
    """Return {strategy: {continuations, stop_reasons, truncated}} for one model.
    Up to three generate passes — natural (min=1) for natural_eos, forced (min=max) for hard_cap,
    criteria-stopped for sentence_stop — each run only if selected in STRATEGIES (GGE_STRATEGIES).
    (``references`` is retained for signature stability — the length_matched oracle that consumed
    it was dropped per REALIGN / DISPATCH #11.)"""
    out = {s: {"continuations": [], "stop_reasons": [], "truncated": []} for s in STRATEGIES}

    for i in range(0, len(prompts), batch_size):
        bp = prompts[i:i + batch_size]
        inputs = tok(bp, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        plens = inputs["attention_mask"].sum(dim=1)

        # Each pass runs ONLY if its strategy is selected (GGE_STRATEGIES) — skipping a pass skips
        # its GPU time and its share of downstream GEC/ERRANT. In full (default) mode the pass
        # order is unchanged, so 3-strategy runs consume the RNG stream exactly as before.
        if "natural_eos" in STRATEGIES:
            nat, nat_len = _gen(model, tok, inputs, plens, min_new=1, max_new=MAX_NEW)
        if "hard_cap" in STRATEGIES:
            cap, _ = _gen(model, tok, inputs, plens, min_new=MAX_NEW, max_new=MAX_NEW)
        if "sentence_stop" in STRATEGIES:
            sent, sent_len = _gen(model, tok, inputs, plens, min_new=1, max_new=MAX_NEW,
                                  criteria=SentenceStopCriteria(tok, plens))

        for b in range(len(bp)):
            if "natural_eos" in STRATEGIES:
                # natural_eos: EOS fired if it stopped before the cap
                out["natural_eos"]["continuations"].append(nat[b])
                eos = nat_len[b] < MAX_NEW
                out["natural_eos"]["stop_reasons"].append("eos" if eos else "cap")
                out["natural_eos"]["truncated"].append(not eos)
            if "hard_cap" in STRATEGIES:
                # hard_cap: forced 64
                out["hard_cap"]["continuations"].append(cap[b])
                out["hard_cap"]["stop_reasons"].append("cap")
                out["hard_cap"]["truncated"].append(True)
            if "sentence_stop" in STRATEGIES:
                # sentence_stop: cut at first terminator; when none appeared, classify by THIS
                # pass's own length (the old code proxied via the natural pass's length, which is
                # unavailable when natural_eos is deselected — and was the wrong pass anyway).
                st = _truncate_at_sentence(sent[b])
                had_end = any(c in sent[b] for c in SENTENCE_ENDERS)
                out["sentence_stop"]["continuations"].append(st)
                out["sentence_stop"]["stop_reasons"].append(
                    "sentence" if had_end else ("eos" if sent_len[b] < MAX_NEW else "cap"))
                out["sentence_stop"]["truncated"].append(had_end)
    return out


# ── GEC + ERRANT for one source's continuations → a raw_results block ───────

def _annotate_block(corrector, ann_cfg: AnnotationConfig, continuations, prompts, references,
                    perplexities, stop_reasons=None, truncated=None) -> dict:
    full_texts = [f"{p} {c}" for p, c in zip(prompts, continuations)]
    res = {
        "continuations": continuations,
        "full_texts": full_texts,
        "perplexities": perplexities,
        "prompt_boundaries": [len(p) for p in prompts],
    }
    res["corrected_continuations"] = correct_in_batches(corrector, continuations, 32, "src:cont")
    res["corrected_full_texts"] = correct_in_batches(corrector, full_texts, 32, "src:full")
    run_annotation(ann_cfg, res)
    classify_errors_by_region(res["full_text_annotations"], res["prompt_boundaries"])
    res["region_error_summary"] = summarize_errors_by_region(res["full_text_annotations"])
    block = {
        "continuations": res["continuations"], "full_texts": res["full_texts"],
        "corrected_continuations": res["corrected_continuations"],
        "corrected_full_texts": res["corrected_full_texts"],
        "perplexities": res["perplexities"], "prompt_boundaries": res["prompt_boundaries"],
        "error_summary": res.get("error_summary", {}),
        "annotations": _serialize_annotations(res.get("annotations", [])),
        "full_text_annotations": _serialize_annotations(res.get("full_text_annotations", [])),
        "full_text_error_summary": res.get("full_text_error_summary", {}),
        "region_error_summary": res["region_error_summary"],
    }
    if stop_reasons is not None:
        block["stop_reasons"] = stop_reasons
        block["truncated"] = truncated
    return block


def _learner_profile(learner_block: dict) -> dict:
    rs = learner_block.get("region_error_summary", {})
    cs = learner_block.get("error_summary", {})
    return {
        "source": "actual_learner_reference_continuation",
        "n_sentences": cs.get("total_sentences", 0),
        "generation_region_error_type_counts": rs.get("generation_error_type_counts", {}),
        "generation_region_total_errors": rs.get("generation_total_errors", 0),
        "continuation_only_error_type_counts": cs.get("error_type_counts", {}),
        "continuation_only_total_errors": cs.get("total_errors", 0),
        "continuation_only_avg_errors_per_sentence": cs.get("avg_errors_per_sentence", 0),
    }


# ── real-fine-tune verification (guard against a base-model fallback) ────────

@torch.no_grad()
def verify_finetune(al_path: str, control_hf_id: str, *, eps: float = 1e-3) -> dict:
    """Prove the AL checkpoint is a GENUINE fine-tune of its matched control — not a
    silent base-model fallback (the broker resolving to the wrong path, an empty
    checkpoint, or the control id itself). Loads both on CPU, compares every
    shared-shape parameter tensor, and returns a verdict.

    A real fine-tune moves the weights: ``total_abs_diff`` is clearly non-zero and
    many parameters change. A base fallback would leave the AL bit-identical to the
    control (``total_abs_diff ≈ 0``). Param-diff is the concrete "the weights are
    different, not just the path existed" check the spec-85 acceptance requires; it
    runs on CPU so it never competes with the GPU generation pass. Raises
    ``AssertionError`` if the AL is indistinguishable from its control.
    """
    from transformers import AutoModelForCausalLM

    al = AutoModelForCausalLM.from_pretrained(al_path, torch_dtype=torch.float32).eval()
    al_sd = {k: v.detach().cpu() for k, v in al.state_dict().items()}
    del al
    gc.collect()
    ctrl = AutoModelForCausalLM.from_pretrained(control_hf_id, torch_dtype=torch.float32).eval()

    n_compared = n_changed = 0
    total_abs = max_abs = 0.0
    for k, cv in ctrl.state_dict().items():
        av = al_sd.get(k)
        if av is None or av.shape != cv.shape:
            continue
        d = (av.float() - cv.float()).abs()
        n_compared += 1
        s, m = float(d.sum()), float(d.max())
        total_abs += s
        if m > eps:
            n_changed += 1
        if m > max_abs:
            max_abs = m
    del ctrl, al_sd
    gc.collect()

    genuine = n_compared > 0 and total_abs > eps and n_changed > 0
    verdict = {
        "al_path": str(al_path), "control": control_hf_id,
        "n_params_compared": n_compared, "n_params_changed": n_changed,
        "total_abs_diff": total_abs, "mean_abs_diff": total_abs / max(1, n_compared),
        "max_abs_diff": max_abs, "genuine_finetune": bool(genuine),
    }
    if not genuine:
        raise AssertionError(
            f"AL at {al_path} is INDISTINGUISHABLE from control {control_hf_id} "
            f"(n_compared={n_compared}, total_abs_diff={total_abs:.3e}) — this looks "
            f"like a base-model fallback, NOT a fine-tune. Refusing to run the matrix."
        )
    return verdict


# ── orchestration ───────────────────────────────────────────────────────────

_KNOWN_FAMILIES = {"gpt2", "smollm2", "pythia"}


def _family_out_base(pairs, model_cfgs: Dict[str, ModelConfig], out_root: str) -> Path:
    """Output subdir derived from the AL side of the first pair, so a smollm2 run
    writes ``smollm2-strategies/`` and never overwrites the gpt2 dead-heat dirs.
    Falls back to ``gpt2-strategies`` only when the family is missing/unknown."""
    al_key = next(iter(pairs))[0] if pairs else None
    al_family = model_cfgs[al_key].model_family if al_key in model_cfgs else None
    family = al_family if al_family in _KNOWN_FAMILIES else "gpt2"
    return Path(out_root) / f"{family}-strategies"


def run_experiment(pairs, model_paths: Dict[str, str], model_cfgs: Dict[str, ModelConfig],
                   dataset_path: str, out_root: str, *, max_sentences=20, seed: int = 42,
                   item_range=None, gec_model_id="grammarly/coedit-large", device=None,
                   log=print) -> dict:
    """Run every (model × strategy), reuse two_signal per (strategy, pair). Returns an aggregate.

    ``model_paths`` maps a registry key → a local model path (AL checkpoints resolved by the
    broker; controls = their HF id). Writes one run dir per strategy under ``out_root``.
    ``seed`` makes the (sampled, do_sample=True) generation reproducible — re-applied before EACH
    source so a source's continuations do not depend on how many other sources ran first.
    ``item_range=(lo, hi)`` slices the loader's (deterministic) item list to one shard — the
    Colab-disconnect insurance for full-corpus runs: each shard writes its own run dir, so a lost
    session costs one shard, not the run. Use with max_sentences=None so every shard slices the
    same full list; point each shard at its OWN out_root (the caller owns dir naming)."""
    device = device or get_device("auto")
    dl = DataLoaderConfig(data_path=dataset_path, text_column="text", max_sentences=max_sentences,
                          min_words=10, max_words=500, prompt_ratio=0.5, min_prompt_words=5)
    items = run_data_loader(dl)
    if item_range is not None:
        lo, hi = int(item_range[0]), int(item_range[1])
        if not (0 <= lo < hi):
            raise ValueError(f"item_range must satisfy 0 <= lo < hi, got {item_range}")
        n_full = len(items)
        items = items[lo:hi]
        if not items:
            raise ValueError(f"item_range {lo}:{hi} selects 0 of {n_full} items — nothing to run")
        log(f"[strategies] item_range {lo}:{hi} -> {len(items)} of {n_full} items")
    prompts = [it["prompt"] for it in items]
    references = [it["reference"] for it in items]
    log(f"[strategies] {len(items)} items; device={device}; strategies={STRATEGIES}")

    sources = sorted({k for pair in pairs for k in pair})   # all AL + control keys

    # 1) generation for every source (model loaded once → all strategies → freed)
    gen_by_source: Dict[str, Dict[str, dict]] = {}
    ppl_by_source: Dict[str, Dict[str, list]] = {}
    for key in sources:
        mc = model_cfgs[key]
        mc = ModelConfig(name=mc.name, hf_model_id=model_paths.get(key, mc.hf_model_id),
                         model_family=mc.model_family, is_learner_tuned=mc.is_learner_tuned)
        log(f"[strategies] generating: {key} (seed={seed})")
        torch.manual_seed(seed)                       # reproducible sampled generation, per source
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        model, tok = load_model(mc, device)
        gen = generate_all_strategies(model, tok, prompts, references, device,
                                      batch_size=model_cfgs[key].batch_size)
        ppl = {}
        for s in STRATEGIES:
            fts = [f"{p} {c}" for p, c in zip(prompts, gen[s]["continuations"])]
            ppl[s] = compute_perplexity(model, tok, fts, batch_size=16, device=device)
        gen_by_source[key], ppl_by_source[key] = gen, ppl
        del model, tok
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 2) GEC + ERRANT for each (source, strategy) + the learner (once) → per-strategy run dirs
    gec_cfg = GECConfig(method="dedicated", model_id=gec_model_id, batch_size=32, device="auto")
    corrector = load_gec_corrector(gec_cfg, device)
    ann_cfg = AnnotationConfig(lang="en")
    log("[strategies] GEC corrector loaded; annotating…")

    learner_block = _annotate_block(corrector, ann_cfg, references, prompts, references,
                                    [0.0] * len(items))   # authentic-learner yardstick (fixed across strategies)
    lp = _learner_profile(learner_block)

    out_base = _family_out_base(pairs, model_cfgs, out_root)
    log(f"[strategies] out_base={out_base}")
    strat_dirs = {}
    for s in STRATEGIES:
        raw = {"learner_baseline": learner_block}
        for key in sources:
            raw[key] = _annotate_block(corrector, ann_cfg, gen_by_source[key][s]["continuations"],
                                       prompts, references, ppl_by_source[key][s],
                                       gen_by_source[key][s]["stop_reasons"],
                                       gen_by_source[key][s]["truncated"])
        d = out_base / s
        d.mkdir(parents=True, exist_ok=True)
        json.dump(raw, open(d / "raw_results.json", "w"), indent=2, default=str)
        json.dump(items, open(d / "prompts.json", "w"), indent=2)
        json.dump(lp, open(d / "learner_profile.json", "w"), indent=2)
        strat_dirs[s] = d
        log(f"[strategies] wrote {s} run dir → {d}")

    # 3) two_signal per (strategy, pair) — reuse the proven Tier-1 analysis wholesale
    grid = []
    for s in STRATEGIES:
        for al, ctrl in pairs:
            ana = strat_dirs[s] / "_analysis" / f"{al}__{ctrl}"
            subprocess.check_call([sys.executable, "-m", "scripts.analysis.two_signal",
                                   "--run-dirs", str(strat_dirs[s]),
                                   "--pair", f"{al}:{ctrl}", "--learner-key", "learner_baseline",
                                   "--out", str(ana)])
            mp = ana / "matched_pairs.tsv"
            row = _read_matched_pairs(mp)
            row.update(strategy=s, pair=f"{al}:{ctrl}", model=al.replace("ft-", ""))
            # per-source length_distance lives in distance_plane.json (not the tsv)
            dp = json.load(open(ana / "distance_plane.json"))
            pts = dp.get("points", {})
            row["ld_al"] = pts.get("artificial_learner", {}).get("length_distance")
            row["ld_control"] = pts.get("matched_control", {}).get("length_distance")
            # stop_reason distribution + %stopped-naturally for the AL under this strategy
            sr = gen_by_source[al][s]["stop_reasons"]
            row["pct_natural_stop"] = round(100 * sum(1 for r in sr if r in ("eos", "sentence")) / max(1, len(sr)), 1)
            row["stop_dist"] = {r: sr.count(r) for r in set(sr)}
            grid.append(row)
            log(f"[strategies] two_signal {s} {al}:{ctrl} → {ana}")
    result = {"grid": grid, "strategy_dirs": {s: str(d) for s, d in strat_dirs.items()},
              "n_items": len(items), "n_scored": len(items), "seed": seed, "gec_model_id": gec_model_id}
    json.dump(_json_safe(result), open(out_base / "strategy_grid.json", "w"), indent=2, default=str)
    return result


def _read_matched_pairs(tsv: Path) -> dict:
    """One matched_pairs.tsv row → dict of its numeric fields (header-driven)."""
    lines = [ln for ln in tsv.read_text().splitlines() if ln.strip()]
    if len(lines) < 2:
        return {}
    hdr = lines[0].split("\t")
    vals = lines[1].split("\t")
    row = {}
    for k, v in zip(hdr, vals):
        if v == "":
            row[k] = None            # empty cell (e.g. a degenerate acq-ratio) → null, not ""
            continue
        try:
            f = float(v)
            row[k] = f if math.isfinite(f) else None   # inf/nan (x/0, 0/0) → null, honest "undefined"
        except ValueError:
            row[k] = v
    return row


def _json_safe(obj):
    """Recursively replace non-finite floats (inf/nan) with None so json.dump emits VALID JSON
    (Python's json writes Infinity/NaN, which are not JSON and break strict parsers)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


# ── rendering (figures / table / summary) ───────────────────────────────────

def _grid_get(grid, strategy, pair, key, default=None):
    for r in grid:
        if r.get("strategy") == strategy and r.get("pair") == pair:
            return r.get(key, default)
    return default


def _best_error_type_strategy(grid, pair, n_items):
    """Closest-to-learner strategy by error-tag JSD — but return an explicit UNDETERMINED
    verdict when the JSD signal is unreliable rather than ranking machine-epsilon noise:
      * too few scored sentences to populate the tag distributions (``n_items < JSD_MIN_N``), or
      * every comparable strategy pinned at the ceiling (sparse support ⇒ JSD saturates to 1.0,
        so the yardstick and the source share almost no tag mass — e.g. learner = {R:DET:1} at n=2).
    Root fix for the confirmatory run: coarsen ERRANT tags / use kl_smoothed at full n."""
    vals = [(s, _grid_get(grid, s, pair, "jsd_al")) for s in COMPARABLE_STRATEGIES]
    valid = [(s, j) for s, j in vals if j is not None]
    if not valid or (n_items or 0) < JSD_MIN_N or all(abs(j - JSD_CEIL) < 1e-6 for _, j in valid):
        return "UNDETERMINED (jsd saturated / n too small)"
    return min(valid, key=lambda sj: sj[1])[0]


def make_figures(result: dict, out_dir: str, pairs) -> dict:
    """Fig 1 length-by-strategy×model, Fig 2 distance-plane small-multiples per strategy,
    a strategy×model grid table (TSV), and per-model closest-strategy summary lines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = result["grid"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    models = [al.replace("ft-", "") for al, _ in pairs]
    pair_by_model = {al.replace("ft-", ""): f"{al}:{ctrl}" for al, ctrl in pairs}

    # Fig 1 — length_ratio by strategy × model (AL), learner reference line at 1.0
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(STRATEGIES))
    w = 0.8 / max(1, len(models))
    for mi, m in enumerate(models):
        vals = [_grid_get(grid, s, pair_by_model[m], "length_ratio_al", 0) or 0 for s in STRATEGIES]
        ax.bar([xi + mi * w for xi in x], vals, w, label=m)
    ax.axhline(1.0, ls="--", c="k", lw=1, label="authentic learner = 1.0")
    ax.set_xticks([xi + w * (len(models) - 1) / 2 for xi in x]); ax.set_xticklabels(STRATEGIES, rotation=15)
    ax.set_ylabel("length_ratio (AL / learner)"); ax.set_title("Over-generation by stopping strategy × model")
    ax.legend(fontsize=8)
    fig.tight_layout(); paths["fig_length"] = str(out / "fig_length_by_strategy.png")
    fig.savefig(paths["fig_length"], dpi=130); plt.close(fig)

    # Fig 2 — distance-to-learner plane, one small-multiple per strategy (x=length_distance, y=JSD)
    fig, axes = plt.subplots(1, len(STRATEGIES), figsize=(4 * len(STRATEGIES), 4), squeeze=False)
    for si, s in enumerate(STRATEGIES):
        ax = axes[0][si]
        for m in models:
            p = pair_by_model[m]
            lx, ly = _grid_get(grid, s, p, "ld_al"), _grid_get(grid, s, p, "jsd_al")
            cx, cy = _grid_get(grid, s, p, "ld_control"), _grid_get(grid, s, p, "jsd_control")
            if None in (lx, ly, cx, cy):
                continue
            ax.annotate("", xy=(lx, ly), xytext=(cx, cy), arrowprops=dict(arrowstyle="->", color="gray"))
            ax.scatter([cx], [cy], marker="s", label=f"{m} control"); ax.scatter([lx], [ly], marker="o", label=f"{m} AL")
        ax.scatter([0], [0], marker="*", s=140, c="k"); ax.set_title(s, fontsize=10)
        ax.set_xlabel("length_distance"); ax.set_ylabel("JSD-to-learner" if si == 0 else "")
        if si == 0:
            ax.legend(fontsize=7)
    fig.suptitle("Distance to the authentic learner (origin) — per strategy; arrow control→AL")
    fig.tight_layout(); paths["fig_planes"] = str(out / "fig_distance_planes.png")
    fig.savefig(paths["fig_planes"], dpi=130); plt.close(fig)

    # Coverage travels WITH the numbers so a metric is never read at the wrong denominator.
    n_items = result.get("n_items", 0)
    n_scored = result.get("n_scored", n_items)

    # Table — strategy × model grid (TSV). `role` is kept for schema stability (every row is now
    # a genuine "strategy" — the length_matched oracle was dropped, DISPATCH #11);
    # n_items/n_scored make the coverage explicit on every row.
    cols = ["strategy", "role", "model", "n_items", "n_scored",
            "length_ratio_al", "jsd_al", "rdr", "pct_natural_stop", "ppl_al"]
    rows = ["\t".join(cols)]
    for s in STRATEGIES:
        role = "oracle" if s in ORACLE_STRATEGIES else "strategy"
        for m in models:
            p = pair_by_model[m]
            cell = {"strategy": s, "role": role, "model": m, "n_items": n_items, "n_scored": n_scored}
            rows.append("\t".join(str(cell[c] if c in cell else _grid_get(grid, s, p, c)) for c in cols))
    paths["table"] = str(out / "strategy_model_grid.tsv")
    Path(paths["table"]).write_text("\n".join(rows) + "\n")

    # Summary — closest-to-learner strategy per model, on each signal + jointly.
    # Ranked over COMPARABLE_STRATEGIES (== all strategies now that the length_matched oracle is
    # dropped — DISPATCH #11); every strategy is a genuine, comparable stopping rule.
    summary = []
    for m in models:
        p = pair_by_model[m]
        best_len = min(COMPARABLE_STRATEGIES, key=lambda s: abs((_grid_get(grid, s, p, "ld_al") or 9)))
        best_jsd = _best_error_type_strategy(grid, p, n_items)   # UNDETERMINED when JSD is saturated / n too small
        if best_jsd.startswith("UNDETERMINED"):
            best_joint = f"{best_len} (length only; error-type undetermined)"
        else:
            best_joint = min(COMPARABLE_STRATEGIES, key=lambda s: (abs(_grid_get(grid, s, p, "ld_al") or 9)
                                                                   + (_grid_get(grid, s, p, "jsd_al") or 9)))
        line = (f"{m}: closest-to-learner strategy — length={best_len}, error-type={best_jsd}, "
                f"joint={best_joint}")
        summary.append(line)
    # Coverage header first — the reader sees the n before the ranking, so an S0 smoke can never
    # be mistaken for the full-corpus result.
    header = f"[n={n_items} sentences (n_scored={n_scored}) — S0 smoke unless n is the full corpus]"
    summary = [header] + summary
    paths["summary"] = str(out / "closest_strategy_summary.txt")
    Path(paths["summary"]).write_text("\n".join(summary) + "\n")
    paths["summary_lines"] = summary
    return paths
