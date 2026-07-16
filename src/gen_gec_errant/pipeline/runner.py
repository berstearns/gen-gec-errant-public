"""Pipeline runner: orchestrates all stages end-to-end with intermediate checkpoints."""

import gc
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from gen_gec_errant.pipeline.config import PipelineConfig, get_model_configs
from gen_gec_errant.data_loader.runner import run_data_loader
from gen_gec_errant.data_loader.config import DataLoaderConfig
from gen_gec_errant.generation.config import GenerationConfig, ModelConfig, GenerationParams
from gen_gec_errant.generation.runner import (
    get_device,
    load_model,
    generate_continuations,
    compute_perplexity,
)
from gen_gec_errant.gec.runner import run_gec, load_gec_corrector, correct_in_batches
from gen_gec_errant.gec.config import GECConfig
from gen_gec_errant.annotation.runner import (
    run_annotation,
    classify_errors_by_region,
    summarize_errors_by_region,
)
from gen_gec_errant.annotation.config import AnnotationConfig
from gen_gec_errant.analysis.runner import run_analysis
from gen_gec_errant.analysis.config import AnalysisConfig

logger = logging.getLogger(__name__)

# ── Intermediate checkpoint helpers ──────────────────────────────────────────

INTERMEDIATE_DIR = "intermediate"
STEP_FILES = {
    1: "step1_items.json",
    2: "step2_generation.json",
    3: "step3_gec.json",
    4: "step4_annotation.json",
}


def _intermediate_path(output_dir: str, step: int) -> Path:
    return Path(output_dir) / INTERMEDIATE_DIR / STEP_FILES[step]


def _step_is_complete(output_dir: str, step: int) -> bool:
    path = _intermediate_path(output_dir, step)
    return path.exists() and path.stat().st_size > 0


def _find_latest_checkpoint(output_dir: str) -> int:
    for step in sorted(STEP_FILES.keys(), reverse=True):
        if _step_is_complete(output_dir, step):
            return step
    return 0


def _save_checkpoint(output_dir: str, step: int, data: object) -> None:
    path = _intermediate_path(output_dir, step)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    tmp.rename(path)
    logger.info("Saved checkpoint: %s", path)


def _load_checkpoint(output_dir: str, step: int) -> object:
    path = _intermediate_path(output_dir, step)
    logger.info("Loading checkpoint: %s", path)
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Corrupt checkpoint %s: %s — will re-run step", path, e)
        return None


# ── Annotation serialization ─────────────────────────────────────────────────

def _serialize_annotations(annotations):
    """Convert annotations to dicts. Handles both dataclass objects and pre-serialized dicts."""
    result = []
    for a in annotations:
        if isinstance(a, dict):
            result.append(a)
            continue
        result.append({
            "original": a.original,
            "corrected": a.corrected,
            "num_errors": a.num_errors,
            "error_type_counts": a.error_type_counts,
            "prompt_error_count": a.prompt_error_count,
            "generation_error_count": a.generation_error_count,
            "prompt_error_type_counts": a.prompt_error_type_counts,
            "generation_error_type_counts": a.generation_error_type_counts,
            "errors": [
                {
                    "orig_tokens": e.original_tokens,
                    "corr_tokens": e.corrected_tokens,
                    "type": e.error_type,
                    "char_start": e.char_start,
                    "char_end": e.char_end,
                    "region": e.region,
                }
                for e in a.errors
            ],
        })
    return result


def _serialize_all_results(all_results: Dict[str, dict]) -> Dict[str, dict]:
    """Serialize all_results to JSON-safe dicts, handling annotation objects."""
    serialized = {}
    for model_name, results in all_results.items():
        entry = {}
        for k, v in results.items():
            if k in ("annotations", "full_text_annotations"):
                entry[k] = _serialize_annotations(v)
            else:
                entry[k] = v
        serialized[model_name] = entry
    return serialized


# ── Pipeline steps ───────────────────────────────────────────────────────────

def _step_1_load_data(config: PipelineConfig) -> List[dict]:
    """Load and prepare data."""
    logger.info("=" * 60)
    logger.info("STEP 1: Loading data")
    logger.info("=" * 60)

    items = run_data_loader(config.data_loader)

    logger.info("Prepared %d prompt-reference pairs", len(items))
    if items:
        logger.info("  Example prompt:    '%s'", items[0]["prompt"])
        logger.info("  Example reference: '%s'", items[0]["reference"])

    return items


def _step_2_generate(
    config: PipelineConfig,
    items: List[dict],
    device: torch.device,
) -> Dict[str, dict]:
    """Generate continuations for all models."""
    logger.info("=" * 60)
    logger.info("STEP 2: Generating text with models")
    logger.info("=" * 60)

    gen_params = config.generation
    model_configs = get_model_configs(config)
    prompts = [item["prompt"] for item in items]
    all_results: Dict[str, dict] = {}

    for mc in model_configs:
        t0 = time.time()

        # Use per-model batch_size if set, otherwise fall back to global
        gen_batch = mc.batch_size if mc.batch_size != 8 else config.batch_size
        logger.info("Using generation batch_size=%d for %s", gen_batch, mc.name)

        model, tokenizer = load_model(mc, device)

        logger.info("Generating with %s...", mc.name)
        continuations = generate_continuations(
            model, tokenizer, prompts, gen_params,
            batch_size=gen_batch, device=device,
        )

        full_texts = [f"{p} {c}" for p, c in zip(prompts, continuations)]

        if device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info("Computing perplexity for %s...", mc.name)
        ppl_batch = min(gen_batch, 32)
        perplexities = compute_perplexity(
            model, tokenizer, full_texts,
            batch_size=ppl_batch, device=device,
        )

        elapsed = time.time() - t0
        logger.info("  %s: generated %d sentences in %.1fs", mc.name, len(continuations), elapsed)
        logger.info("  Mean perplexity: %.2f", sum(perplexities) / len(perplexities))

        all_results[mc.name] = {
            "continuations": continuations,
            "full_texts": full_texts,
            "perplexities": perplexities,
            "prompt_boundaries": [len(p) for p in prompts],
        }

        del model, tokenizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return all_results


def _step_3_gec(config: PipelineConfig, all_results: Dict[str, dict]) -> Dict[str, dict]:
    """Run GEC on all generated text — loads GEC model once for all models."""
    logger.info("=" * 60)
    logger.info("STEP 3: Grammatical Error Correction")
    logger.info("=" * 60)

    gec_config = config.gec
    device = get_device(gec_config.device)
    corrector = load_gec_corrector(gec_config, device)

    for model_name, results in all_results.items():
        logger.info("Correcting %s outputs...", model_name)

        corrected_continuations = correct_in_batches(
            corrector, results["continuations"], gec_config.batch_size,
            f"{model_name}:continuations")
        corrected_full_texts = correct_in_batches(
            corrector, results["full_texts"], gec_config.batch_size,
            f"{model_name}:full_texts")

        results["corrected_continuations"] = corrected_continuations
        results["corrected_full_texts"] = corrected_full_texts
        logger.info("Corrected %d sentences for %s", len(corrected_continuations), model_name)

    del corrector
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return all_results


def _add_learner_baseline(items: List[dict], all_results: Dict[str, dict]) -> None:
    """Add learner baseline as a pseudo-model entry for comparison."""
    logger.info("Adding learner baseline...")
    all_results["learner_baseline"] = {
        "full_texts": [item["full"] for item in items],
        "continuations": [item["reference"] for item in items],
        "perplexities": [0.0] * len(items),
        "prompt_boundaries": [len(item["prompt"]) for item in items],
    }


def _step_4_annotate(config: PipelineConfig, all_results: Dict[str, dict]) -> Dict[str, dict]:
    """Run ERRANT annotation."""
    logger.info("=" * 60)
    logger.info("STEP 4: ERRANT Error Annotation")
    logger.info("=" * 60)

    ann_config = config.annotation

    for model_name, results in all_results.items():
        logger.info("Annotating errors for %s...", model_name)
        run_annotation(ann_config, results)

        # Region classification on full-text annotations
        if "full_text_annotations" in results and "prompt_boundaries" in results:
            classify_errors_by_region(results["full_text_annotations"], results["prompt_boundaries"])
            results["region_error_summary"] = summarize_errors_by_region(results["full_text_annotations"])

    return all_results


def _write_learner_profile(output_dir: Path, learner_results: dict) -> None:
    """Extract the actual-learner error-tag profile (G3 leg) into its own artifact.

    `learner_results` is the "learner_baseline" pseudo-model entry: the actual
    learner's reference continuation, run through the identical GEC+ERRANT
    instrument as every generator (see `_add_learner_baseline`). Per
    30-data-integrity-rules.md #4, only generation-region errors (the
    continuation, not the shared prompt) count as the learner's own — that
    distribution is what 20-preregistration.md's JSD compares each model against.
    """
    region_summary = learner_results.get("region_error_summary", {})
    continuation_summary = learner_results.get("error_summary", {})

    profile = {
        "source": "actual_learner_reference_continuation",
        "n_sentences": continuation_summary.get("total_sentences", 0),
        "generation_region_error_type_counts": region_summary.get("generation_error_type_counts", {}),
        "generation_region_total_errors": region_summary.get("generation_total_errors", 0),
        "continuation_only_error_type_counts": continuation_summary.get("error_type_counts", {}),
        "continuation_only_total_errors": continuation_summary.get("total_errors", 0),
        "continuation_only_avg_errors_per_sentence": continuation_summary.get("avg_errors_per_sentence", 0),
    }

    path = output_dir / "learner_profile.json"
    with open(path, "w") as f:
        json.dump(profile, f, indent=2)
    logger.info("Saved actual-learner error-tag profile to %s", path)


def _step_5_analyze(
    config: PipelineConfig,
    all_results: Dict[str, dict],
    items: List[dict],
) -> Tuple[list, dict]:
    """Compute summaries, comparisons, plots, CSVs."""
    logger.info("=" * 60)
    logger.info("STEP 5: Analysis & Visualization")
    logger.info("=" * 60)

    analysis_config = config.analysis
    analysis_config.output_dir = config.output_dir
    analysis_config.skip_plots = config.skip_plots

    # Save raw data for reproducibility
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_output = {}
    for model_name, results in all_results.items():
        raw_output[model_name] = {
            "continuations": results["continuations"],
            "full_texts": results["full_texts"],
            "corrected_continuations": results.get("corrected_continuations", []),
            "corrected_full_texts": results.get("corrected_full_texts", []),
            "perplexities": results["perplexities"],
            "prompt_boundaries": results.get("prompt_boundaries", []),
            "error_summary": results.get("error_summary", {}),
            "annotations": _serialize_annotations(results.get("annotations", [])),
            "full_text_annotations": _serialize_annotations(
                results.get("full_text_annotations", [])),
            "full_text_error_summary": results.get("full_text_error_summary", {}),
            "region_error_summary": results.get("region_error_summary", {}),
        }

    with open(output_dir / "raw_results.json", "w") as f:
        json.dump(raw_output, f, indent=2, default=str)
    logger.info("Saved raw results to %s", output_dir / "raw_results.json")

    with open(output_dir / "prompts.json", "w") as f:
        json.dump(items, f, indent=2)

    if "learner_baseline" in all_results:
        _write_learner_profile(output_dir, all_results["learner_baseline"])

    return run_analysis(analysis_config, all_results, items)


def _load_raw_results(output_dir: str, items: List[dict]) -> Dict[str, dict]:
    """Load raw_results.json and reconstruct missing keys."""
    raw_path = Path(output_dir) / "raw_results.json"
    logger.info("Loading existing results from %s", raw_path)
    with open(raw_path) as f:
        all_results = json.load(f)

    prompts = [item["prompt"] for item in items]

    for model_name, results in all_results.items():
        if "full_texts" not in results:
            results["full_texts"] = [
                f"{p} {c}" for p, c in zip(prompts, results["continuations"])
            ]
        if "corrected_full_texts" not in results:
            results["corrected_full_texts"] = []

    return all_results


# ── Main entry point ─────────────────────────────────────────────────────────

def run_pipeline(config: PipelineConfig) -> Tuple[list, dict]:
    """
    Run the full pipeline end-to-end with intermediate checkpoints.

    When config.resume is True, detects the last completed step and
    resumes from there, skipping already-completed work.

    Returns:
        (summaries, comparison)
    """
    if config.skip_gec:
        config.skip_generation = True

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = get_device(config.device)
    logger.info("Using device: %s", device)

    # Determine resume point
    latest = 0
    if config.resume:
        latest = _find_latest_checkpoint(config.output_dir)
        if latest > 0:
            logger.info("Resume: found checkpoint at step %d, will skip steps 1–%d", latest, latest)
        else:
            logger.info("Resume: no checkpoints found, starting from scratch")

    # ── Step 1: Load data ──
    if latest >= 1:
        items = _load_checkpoint(config.output_dir, 1)
        if items is None:
            items = _step_1_load_data(config)
            _save_checkpoint(config.output_dir, 1, items)
        else:
            logger.info("Resumed step 1: %d items from checkpoint", len(items))
    else:
        items = _step_1_load_data(config)
        _save_checkpoint(config.output_dir, 1, items)

    # ── Step 2: Generate ──
    if latest >= 2:
        all_results = _load_checkpoint(config.output_dir, 2)
        if all_results is None:
            latest = 1  # corrupt checkpoint, re-run from here
        else:
            logger.info("Resumed step 2: %d models from checkpoint", len(all_results))
    if latest < 2:
        if config.skip_generation:
            all_results = _load_raw_results(config.output_dir, items)
        else:
            all_results = _step_2_generate(config, items, device)
            if config.include_learner_baseline:
                _add_learner_baseline(items, all_results)
            _save_checkpoint(config.output_dir, 2, all_results)

    # Ensure learner baseline is present
    if config.include_learner_baseline and "learner_baseline" not in all_results:
        _add_learner_baseline(items, all_results)

    # Free GPU before GEC
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        free_mb = torch.cuda.mem_get_info()[0] // (1024 * 1024)
        logger.info("GPU memory free before GEC: %d MiB", free_mb)

    # ── Step 3: GEC ──
    if latest >= 3:
        all_results = _load_checkpoint(config.output_dir, 3)
        if all_results is None:
            latest = 2
        else:
            logger.info("Resumed step 3: GEC results from checkpoint")
    if latest < 3:
        if not config.skip_gec:
            all_results = _step_3_gec(config, all_results)
        _save_checkpoint(config.output_dir, 3, all_results)

    # ── Step 4: Annotate ──
    if latest >= 4:
        all_results = _load_checkpoint(config.output_dir, 4)
        if all_results is None:
            latest = 3
        else:
            logger.info("Resumed step 4: annotation results from checkpoint")
    if latest < 4:
        all_results = _step_4_annotate(config, all_results)
        _save_checkpoint(config.output_dir, 4, _serialize_all_results(all_results))

    # ── Step 5: Analyze ──
    summaries, comparison = _step_5_analyze(config, all_results, items)

    logger.info("Pipeline complete!")
    return summaries, comparison
