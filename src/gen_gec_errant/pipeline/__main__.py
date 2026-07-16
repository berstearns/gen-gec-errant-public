"""CLI entry point: python -m gen_gec_errant.pipeline"""

import argparse
import logging

from gen_gec_errant.pipeline.config import PipelineConfig, load_config_from_yaml, apply_cli_overrides
from gen_gec_errant.pipeline.runner import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full gen-gec-errant pipeline",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--skip_gec", action="store_true")
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last completed intermediate checkpoint")
    return parser


def main(argv: list[str] | None = None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = build_parser()
    args, remaining = parser.parse_known_args(argv)

    if args.config:
        config = load_config_from_yaml(args.config)
    else:
        config = PipelineConfig()

    # Apply explicit CLI args
    for field_name in ("output_dir", "device", "batch_size", "seed"):
        val = getattr(args, field_name, None)
        if val is not None:
            setattr(config, field_name, val)

    if args.skip_generation:
        config.skip_generation = True
    if args.skip_gec:
        config.skip_gec = True
    if args.skip_plots:
        config.skip_plots = True
    if args.resume:
        config.resume = True

    config = apply_cli_overrides(config, remaining)

    run_pipeline(config)


if __name__ == "__main__":
    main()
