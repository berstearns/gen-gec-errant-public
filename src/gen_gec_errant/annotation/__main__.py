"""CLI entry point: python -m gen_gec_errant.annotation"""

import argparse
import json
import logging

from gen_gec_errant.annotation.config import AnnotationConfig, load_config_from_yaml, apply_cli_overrides
from gen_gec_errant.annotation.runner import run_annotation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ERRANT error annotation on GEC results",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--input", type=str, required=True, help="Path to GEC results JSON")
    return parser


def main(argv: list[str] | None = None) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    parser = build_parser()
    args, remaining = parser.parse_known_args(argv)

    if args.config:
        config = load_config_from_yaml(args.config)
    else:
        config = AnnotationConfig()

    config = apply_cli_overrides(config, remaining)

    with open(args.input) as f:
        gec_results = json.load(f)

    results = run_annotation(config, gec_results)

    summary = results.get("error_summary", {})
    print(json.dumps({
        "total_errors": summary.get("total_errors", 0),
        "error_rate": summary.get("error_rate", 0),
        "top_5_types": summary.get("top_10_error_types", [])[:5],
    }, indent=2))
    return results


if __name__ == "__main__":
    main()
