"""Visualization functions for analysis results."""

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def _import_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_perplexity_comparison(summaries: List[Dict], output_path: str):
    """Bar chart of mean perplexity per model."""
    plt = _import_matplotlib()

    names = [s["model_name"] for s in summaries]
    means = [s["ppl_mean"] for s in summaries]
    stds = [s["ppl_std"] for s in summaries]
    colors = ["#2196F3" if "learner" not in n else "#F44336" for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(names, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
    ax.set_ylabel("Mean Perplexity", fontsize=12)
    ax.set_title("Perplexity by Model", fontsize=14)
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved perplexity plot: %s", output_path)


def plot_error_comparison(summaries: List[Dict], output_path: str):
    """Bar chart of average errors per sentence."""
    plt = _import_matplotlib()

    names = [s["model_name"] for s in summaries]
    avg_errors = [s["avg_errors_per_sentence"] for s in summaries]
    error_rates = [s["error_rate"] for s in summaries]
    colors = ["#2196F3" if "learner" not in n else "#F44336" for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.bar(names, avg_errors, color=colors, alpha=0.8)
    ax1.set_ylabel("Avg Errors per Sentence", fontsize=12)
    ax1.set_title("Error Count by Model", fontsize=14)
    ax1.tick_params(axis="x", rotation=30)

    ax2.bar(names, error_rates, color=colors, alpha=0.8)
    ax2.set_ylabel("Fraction of Sentences with Errors", fontsize=12)
    ax2.set_title("Error Rate by Model", fontsize=14)
    ax2.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved error comparison plot: %s", output_path)


def plot_error_type_breakdown(summaries: List[Dict], output_path: str, top_n: int = 10):
    """Grouped bar chart of top N error types across models."""
    plt = _import_matplotlib()

    all_types: Dict[str, int] = {}
    for s in summaries:
        for etype, count in s["error_type_counts"].items():
            all_types[etype] = all_types.get(etype, 0) + count

    top_types = sorted(all_types.items(), key=lambda x: -x[1])[:top_n]
    type_names = [t[0] for t in top_types]

    x = np.arange(len(type_names))
    width = 0.8 / len(summaries)

    fig, ax = plt.subplots(figsize=(14, 7))
    for i, s in enumerate(summaries):
        counts = [s["error_type_counts"].get(t, 0) for t in type_names]
        offset = (i - len(summaries) / 2 + 0.5) * width
        ax.bar(x + offset, counts, width, label=s["model_name"], alpha=0.8)

    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Top {top_n} Error Types by Model", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(type_names, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved error type breakdown plot: %s", output_path)


def plot_ppl_vs_errors_scatter(summaries: List[Dict], output_path: str):
    """Scatter plot: perplexity vs error count per sentence, colored by model."""
    plt = _import_matplotlib()

    fig, ax = plt.subplots(figsize=(10, 8))
    for s in summaries:
        ppls = [x["ppl"] for x in s["per_sentence_ppl_plus_errors"]]
        errs = [x["errors"] for x in s["per_sentence_ppl_plus_errors"]]
        ax.scatter(ppls, errs, alpha=0.3, s=20, label=s["model_name"])

    ax.set_xlabel("Perplexity", fontsize=12)
    ax.set_ylabel("Error Count", fontsize=12)
    ax.set_title("Perplexity vs Grammatical Errors per Sentence", fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved scatter plot: %s", output_path)


def plot_combined_metric(summaries: List[Dict], output_path: str):
    """Bar chart of the combined PPL x Errors metric."""
    plt = _import_matplotlib()

    names = [s["model_name"] for s in summaries]
    combined = [s["ppl_x_errors"] for s in summaries]
    colors = ["#2196F3" if "learner" not in n else "#F44336" for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(names, combined, color=colors, alpha=0.8)
    ax.set_ylabel("Mean PPL x Avg Errors", fontsize=12)
    ax.set_title("Combined Metric: Perplexity x Error Rate", fontsize=14)
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved combined metric plot: %s", output_path)


def generate_all_plots(summaries: List[Dict], output_dir: str, top_n: int = 10):
    """Generate all visualization plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_perplexity_comparison(summaries, str(output_dir / "perplexity_comparison.png"))
    plot_error_comparison(summaries, str(output_dir / "error_comparison.png"))
    plot_error_type_breakdown(summaries, str(output_dir / "error_type_breakdown.png"), top_n=top_n)
    plot_ppl_vs_errors_scatter(summaries, str(output_dir / "ppl_vs_errors_scatter.png"))
    plot_combined_metric(summaries, str(output_dir / "combined_metric.png"))
