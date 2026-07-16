"""CLI entry point: python -m gen_gec_errant.data_loader"""

import argparse
import json
import logging
import sys

from gen_gec_errant.data_loader.config import DataLoaderConfig, load_config_from_yaml, apply_cli_overrides
from gen_gec_errant.data_loader.runner import run_data_loader


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load and prepare data for the gen-gec-errant pipeline",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--max_sentences", type=int, default=None)
    parser.add_argument("--min_words", type=int, default=None)
    parser.add_argument("--max_words", type=int, default=None)
    parser.add_argument("--prompt_ratio", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> list[dict]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    parser = build_parser()
    args, remaining = parser.parse_known_args(argv)

    if args.config:
        config = load_config_from_yaml(args.config)
    else:
        config = DataLoaderConfig()

    # Apply explicit CLI args
    for field_name in ("data_path", "max_sentences", "min_words", "max_words", "prompt_ratio"):
        val = getattr(args, field_name, None)
        if val is not None:
            setattr(config, field_name, val)

    config = apply_cli_overrides(config, remaining)

    items = run_data_loader(config)

    # Print summary
    print(json.dumps({"num_items": len(items), "sample": items[:2]}, indent=2))
    return items


if __name__ == "__main__":
    main()
