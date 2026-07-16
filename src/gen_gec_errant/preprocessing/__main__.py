"""CLI entry point: python -m gen_gec_errant.preprocessing"""

import argparse
import logging

from gen_gec_errant.preprocessing.config import (
    PreprocessingConfig,
    load_config_from_yaml,
    apply_cli_overrides,
)
from gen_gec_errant.preprocessing.runner import run_preprocessing


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess EFCAMDAT CSV into sentence-level data",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--input", "-i", type=str, default=None, help="Input CSV path")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output CSV path")
    parser.add_argument("--min_words", type=int, default=None)
    parser.add_argument("--max_words", type=int, default=None)
    parser.add_argument("--max_essays", type=int, default=None)
    parser.add_argument("--cefr_filter", type=str, default=None)
    parser.add_argument("--l1_filter", type=str, default=None)
    return parser


def main(argv: list[str] | None = None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    parser = build_parser()
    args, remaining = parser.parse_known_args(argv)

    if args.config:
        config = load_config_from_yaml(args.config)
    else:
        config = PreprocessingConfig()

    if args.input:
        config.input_path = args.input
    if args.output:
        config.output_path = args.output
    for field_name in ("min_words", "max_words", "max_essays", "cefr_filter", "l1_filter"):
        val = getattr(args, field_name, None)
        if val is not None:
            setattr(config, field_name, val)

    config = apply_cli_overrides(config, remaining)

    output_path = run_preprocessing(config)
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
