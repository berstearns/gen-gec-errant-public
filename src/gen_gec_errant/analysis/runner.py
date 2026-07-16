"""Analysis runner: compute summaries, comparisons, generate plots and CSVs."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from gen_gec_errant.analysis.config import AnalysisConfig
from gen_gec_errant.analysis.csv_export import export_csv, export_errors_long_format
from gen_gec_errant.analysis.plots import generate_all_plots

logger = logging.getLogger(__name__)


def _import_scipy():
    from scipy import stats
    return stats


def compute_model_summary(
    model_name: str,
    perplexities: List[float],
    error_summaries: Dict,
    annotations: List,
    full_text_error_summary: Optional[Dict] = None,
    region_error_summary: Optional[Dict] = None,
) -> Dict:
    """Compute comprehensive summary metrics for a single model."""
    ppl_array = np.array(perplexities)
    errors_per_sent = np.array(error_summaries["errors_per_sentence"])

    summary = {
        "model_name": model_name,
        "ppl_mean": float(np.mean(ppl_array)),
        "ppl_median": float(np.median(ppl_array)),
        "ppl_std": float(np.std(ppl_array)),
        "ppl_25th": float(np.percentile(ppl_array, 25)),
        "ppl_75th": float(np.percentile(ppl_array, 75)),
        "total_errors": error_summaries["total_errors"],
        "avg_errors_per_sentence": error_summaries["avg_errors_per_sentence"],
        "error_rate": error_summaries["error_rate"],
        "sentences_with_errors": error_summaries["sentences_with_errors"],
        "total_sentences": error_summaries["total_sentences"],
        "top_10_error_types": error_summaries["top_10_error_types"],
        "error_type_counts": error_summaries["error_type_counts"],
        "ppl_x_errors": float(np.mean(ppl_array) * error_summaries["avg_errors_per_sentence"]),
        "per_sentence_ppl_plus_errors": [
            {"ppl": float(p), "errors": int(e)}
            for p, e in zip(ppl_array, errors_per_sent)
        ],
    }

    if full_text_error_summary:
        summary["ft_total_errors"] = full_text_error_summary["total_errors"]
        summary["ft_avg_errors_per_sentence"] = full_text_error_summary["avg_errors_per_sentence"]
        summary["ft_error_rate"] = full_text_error_summary["error_rate"]

    if region_error_summary:
        summary["ft_prompt_errors"] = region_error_summary["prompt_total_errors"]
        summary["ft_generation_errors"] = region_error_summary["generation_total_errors"]

    return summary


def compare_models(summaries: List[Dict]) -> Dict:
    """Compare multiple model summaries with statistical tests."""
    stats_mod = _import_scipy()

    comparison = {
        "models": [s["model_name"] for s in summaries],
        "ppl_means": [s["ppl_mean"] for s in summaries],
        "error_rates": [s["error_rate"] for s in summaries],
        "avg_errors": [s["avg_errors_per_sentence"] for s in summaries],
        "ppl_x_errors": [s["ppl_x_errors"] for s in summaries],
    }

    pairwise = []
    for i in range(len(summaries)):
        for j in range(i + 1, len(summaries)):
            s1, s2 = summaries[i], summaries[j]

            ppls1 = [x["ppl"] for x in s1["per_sentence_ppl_plus_errors"]]
            ppls2 = [x["ppl"] for x in s2["per_sentence_ppl_plus_errors"]]
            errs1 = [x["errors"] for x in s1["per_sentence_ppl_plus_errors"]]
            errs2 = [x["errors"] for x in s2["per_sentence_ppl_plus_errors"]]

            try:
                ppl_stat, ppl_pval = stats_mod.mannwhitneyu(ppls1, ppls2, alternative="two-sided")
            except Exception:
                ppl_stat, ppl_pval = None, None

            try:
                err_stat, err_pval = stats_mod.mannwhitneyu(errs1, errs2, alternative="two-sided")
            except Exception:
                err_stat, err_pval = None, None

            pairwise.append({
                "model_a": s1["model_name"],
                "model_b": s2["model_name"],
                "ppl_test_stat": ppl_stat,
                "ppl_p_value": ppl_pval,
                "error_test_stat": err_stat,
                "error_p_value": err_pval,
            })

    comparison["pairwise_tests"] = pairwise
    return comparison


def save_results(summaries: List[Dict], comparison: Dict, output_dir: str):
    """Save all results to JSON files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for s in summaries:
        path = output_dir / f"{s['model_name']}_summary.json"
        with open(path, "w") as f:
            json.dump(s, f, indent=2, default=str)
        logger.info("Saved: %s", path)

    path = output_dir / "model_comparison.json"
    with open(path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    logger.info("Saved: %s", path)


def run_analysis(
    config: AnalysisConfig,
    all_results: Dict[str, dict],
    items: List[dict],
) -> Tuple[List[Dict], Dict]:
    """
    Run analysis on all model results.

    Args:
        config: AnalysisConfig
        all_results: Dict[model_name -> results dict]
        items: List of data_loader items

    Returns:
        (summaries, comparison)
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute per-model summaries
    summaries = []
    for model_name, results in all_results.items():
        summary = compute_model_summary(
            model_name=model_name,
            perplexities=results["perplexities"],
            error_summaries=results["error_summary"],
            annotations=results["annotations"],
            full_text_error_summary=results.get("full_text_error_summary"),
            region_error_summary=results.get("region_error_summary"),
        )
        summaries.append(summary)

    # Compare models
    comparison = compare_models(summaries)

    # Log comparison table
    logger.info("=" * 110)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 110)
    logger.info(
        "%-20s %10s %10s %10s %10s %12s %10s %10s %10s",
        "Model", "PPL Mean", "PPL Std", "Err Rate", "Avg Err", "PPL*Err",
        "FT Errors", "Prompt Er", "Gen Er",
    )
    logger.info("-" * 110)
    for s in summaries:
        logger.info(
            "%-20s %10.2f %10.2f %10.3f %10.3f %12.2f %10s %10s %10s",
            s["model_name"], s["ppl_mean"], s["ppl_std"],
            s["error_rate"], s["avg_errors_per_sentence"], s["ppl_x_errors"],
            str(s.get("ft_total_errors", "")),
            str(s.get("ft_prompt_errors", "")),
            str(s.get("ft_generation_errors", "")),
        )
    logger.info("=" * 110)

    # Save JSON results
    save_results(summaries, comparison, config.output_dir)

    # Generate plots
    if not config.skip_plots:
        try:
            generate_all_plots(
                summaries,
                str(output_dir / "plots"),
                top_n=config.top_n_error_types,
            )
            logger.info("All plots generated successfully")
        except Exception as e:
            logger.warning("Plot generation failed: %s", e)

    # Export CSVs
    model_names = list(all_results.keys())

    export_csv(
        items=items,
        all_results=all_results,
        model_names=model_names,
        output_path=str(output_dir / "full_results.csv"),
    )

    export_errors_long_format(
        all_results=all_results,
        model_names=model_names,
        output_path=str(output_dir / "errors_long_format.csv"),
        items=items,
    )

    return summaries, comparison
