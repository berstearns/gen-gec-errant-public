"""CLI entry point: python -m gen_gec_errant.gec"""

import argparse
import json
import logging

from gen_gec_errant.gec.config import GECConfig, load_config_from_yaml, apply_cli_overrides
from gen_gec_errant.gec.runner import run_gec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run grammatical error correction on generation results",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--input", type=str, required=True, help="Path to generation results JSON")
    return parser


def main(argv: list[str] | None = None) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    parser = build_parser()
    args, remaining = parser.parse_known_args(argv)

    if args.config:
        config = load_config_from_yaml(args.config)
    else:
        config = GECConfig()

    config = apply_cli_overrides(config, remaining)

    with open(args.input) as f:
        generation_results = json.load(f)

    results = run_gec(config, generation_results)

    print(json.dumps({
        "num_corrected": len(results.get("corrected_continuations", [])),
    }, indent=2))
    return results


if __name__ == "__main__":
    main()
