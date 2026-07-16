"""CLI entry point: python -m gen_gec_errant.generation"""

import argparse
import json
import logging
import sys

from gen_gec_errant.generation.config import GenerationConfig, load_config_from_yaml, apply_cli_overrides
from gen_gec_errant.generation.runner import run_generation
from gen_gec_errant.data_loader.runner import run_data_loader
from gen_gec_errant.data_loader.config import DataLoaderConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate text with a language model",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--data_path", type=str, default=None, help="Input data path (overrides data_loader)")
    parser.add_argument("--max_sentences", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    parser = build_parser()
    args, remaining = parser.parse_known_args(argv)

    if args.config:
        config = load_config_from_yaml(args.config)
    else:
        config = GenerationConfig()

    config = apply_cli_overrides(config, remaining)

    # Load data
    dl_config = DataLoaderConfig()
    if args.data_path:
        dl_config.data_path = args.data_path
    if args.max_sentences:
        dl_config.max_sentences = args.max_sentences
    items = run_data_loader(dl_config)

    results = run_generation(config, items)

    print(json.dumps({
        "model": results["model_name"],
        "num_generated": len(results["continuations"]),
        "mean_ppl": sum(results["perplexities"]) / len(results["perplexities"]),
    }, indent=2))
    return results


if __name__ == "__main__":
    main()
