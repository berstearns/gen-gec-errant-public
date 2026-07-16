"""CLI entry point: python -m gen_gec_errant.analysis"""

import argparse
import json
import logging

from gen_gec_errant.analysis.config import AnalysisConfig, load_config_from_yaml, apply_cli_overrides
from gen_gec_errant.analysis.runner import run_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run analysis on pipeline results",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--input", type=str, required=True, help="Path to raw_results.json")
    parser.add_argument("--items", type=str, required=True, help="Path to prompts.json")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--skip_plots", action="store_true")
    return parser


def main(argv: list[str] | None = None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    parser = build_parser()
    args, remaining = parser.parse_known_args(argv)

    if args.config:
        config = load_config_from_yaml(args.config)
    else:
        config = AnalysisConfig()

    if args.output_dir:
        config.output_dir = args.output_dir
    if args.skip_plots:
        config.skip_plots = True

    config = apply_cli_overrides(config, remaining)

    with open(args.input) as f:
        all_results = json.load(f)

    with open(args.items) as f:
        items = json.load(f)

    summaries, comparison = run_analysis(config, all_results, items)

    print(json.dumps({
        "num_models": len(summaries),
        "models": [s["model_name"] for s in summaries],
    }, indent=2))


if __name__ == "__main__":
    main()
