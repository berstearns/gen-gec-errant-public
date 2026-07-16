#!/usr/bin/env python3
"""Compare original learner texts with model-generated continuations.

Always shows full texts with the prompt portion dimmed and the generated
portion highlighted, so you can see at a glance what the model produced.

Usage:
    # Auto-discover all output dirs
    python scripts/compare_texts.py

    # Specify output dirs explicitly
    python scripts/compare_texts.py outputs/dir-model-a outputs/dir-model-b

    # Dump all examples as markdown (no interactive prompt)
    python scripts/compare_texts.py --all

    # Show only continuations (no prompt) for models
    python scripts/compare_texts.py --continuation-only

    # Wrap text at N columns (default: 80)
    python scripts/compare_texts.py --wrap 120
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

# ANSI escape codes
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"


def load_output_dir(path: Path) -> dict | None:
    """Load prompts and raw_results from an output directory."""
    prompts_file = path / "prompts.json"
    results_file = path / "raw_results.json"
    if not prompts_file.exists() or not results_file.exists():
        return None
    with open(prompts_file) as f:
        prompts = json.load(f)
    with open(results_file) as f:
        results = json.load(f)
    return {"prompts": prompts, "results": results}


def discover_output_dirs(root: Path) -> list[Path]:
    """Find output directories containing prompts.json + raw_results.json."""
    dirs = []
    if not root.exists():
        return dirs
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and (entry / "raw_results.json").exists():
            dirs.append(entry)
    return dirs


def wrap_plain(text: str, width: int) -> list[str]:
    """Wrap text to given width, preserving existing newlines. Returns lines."""
    out = []
    for line in text.splitlines():
        if len(line) <= width:
            out.append(line)
        else:
            out.extend(
                textwrap.wrap(line, width=width,
                              break_long_words=False, break_on_hyphens=False)
                or [""]
            )
    return out


def colorize_wrapped(text: str, boundary: int, width: int,
                     prompt_style: str = DIM,
                     gen_style: str = BOLD + GREEN) -> list[str]:
    """Wrap plain text, then apply prompt/generation colors split at boundary."""
    lines = wrap_plain(text, width)
    colored = []
    pos = 0
    for line in lines:
        line_len = len(line)
        line_end = pos + line_len
        if line_end <= boundary:
            # Entire line is prompt
            colored.append(f"{prompt_style}{line}{RESET}")
        elif pos >= boundary:
            # Entire line is generation
            colored.append(f"{gen_style}{line}{RESET}")
        else:
            # Boundary falls within this line
            split_at = boundary - pos
            colored.append(
                f"{prompt_style}{line[:split_at]}{RESET}"
                f"{gen_style}{line[split_at:]}{RESET}"
            )
        # +1 for the newline/space that wrapping consumed
        pos = line_end + 1
    return colored


def format_example(
    idx: int,
    total: int,
    prompt_text: str,
    full_original: str,
    prompt_len: int,
    model_entries: list[tuple[str, str, int]],  # (name, full_text, boundary)
    wrap_width: int,
) -> str:
    """Format a single example with colored prompt/generation split."""
    lines = []
    lines.append(f"\n{'━' * wrap_width}")
    lines.append(f"  Example {idx + 1} / {total}")
    lines.append(f"{'━' * wrap_width}")

    # Legend
    lines.append(f"\n  {DIM}dim = prompt (shared){RESET}  "
                 f"{BOLD}{GREEN}bold green = generated/continuation{RESET}\n")

    # Original learner text
    sep = "─" * wrap_width
    lines.append(f"{YELLOW}{BOLD}▌ Original (learner){RESET}")
    lines.extend(colorize_wrapped(full_original, prompt_len, wrap_width,
                                  prompt_style=DIM, gen_style=BOLD + CYAN))
    lines.append(sep)

    # Each model
    for name, full_text, boundary in model_entries:
        lines.append(f"{MAGENTA}{BOLD}▌ {name}{RESET}")
        lines.extend(colorize_wrapped(full_text, boundary, wrap_width))
        lines.append(sep)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare original learner texts with model generations"
    )
    parser.add_argument(
        "output_dirs",
        nargs="*",
        help="Output directories to load (default: auto-discover from outputs/)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Print all examples at once (no interactive navigation)",
    )
    parser.add_argument(
        "--continuation-only", action="store_true",
        help="Show only the generated continuation for models (no prompt)",
    )
    parser.add_argument(
        "--wrap", type=int, default=80,
        help="Wrap text at N columns (default: 80)",
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="Start at example N (0-indexed)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    # Resolve output dirs
    if args.output_dirs:
        output_dirs = [Path(d) for d in args.output_dirs]
    else:
        output_dirs = discover_output_dirs(project_root / "outputs")

    if not output_dirs:
        print("No output directories found. Run the pipeline first or specify dirs.",
              file=sys.stderr)
        sys.exit(1)

    # Load all data
    prompts = None
    # model_name -> {full_texts: [...], prompt_boundaries: [...], ...}
    all_models: dict[str, dict] = {}

    for d in output_dirs:
        data = load_output_dir(d)
        if data is None:
            print(f"[skip] {d} (missing prompts.json or raw_results.json)",
                  file=sys.stderr)
            continue

        if prompts is None:
            prompts = data["prompts"]

        for model_name, model_data in data["results"].items():
            if model_name == "learner_baseline":
                continue
            key = model_name
            if key in all_models:
                key = f"{model_name} ({d.name})"
            all_models[key] = model_data

    if prompts is None:
        print("No valid output data found.", file=sys.stderr)
        sys.exit(1)

    n_examples = len(prompts)
    model_names = sorted(all_models.keys())

    print(f"Loaded {len(model_names)} model(s): {', '.join(model_names)}",
          file=sys.stderr)
    print(f"Examples: {n_examples}", file=sys.stderr)
    if not args.all:
        print("Navigation: [Enter]=next  [p]=previous  [N]=jump to N  [q]=quit\n",
              file=sys.stderr)

    # Display loop
    idx = args.start
    while 0 <= idx < n_examples:
        p = prompts[idx]
        prompt_text = p["prompt"]
        full_original = p["full"]
        prompt_len = len(prompt_text)

        model_entries = []
        for name in model_names:
            mdata = all_models[name]
            if args.continuation_only:
                conts = mdata.get("continuations", [])
                text = conts[idx] if idx < len(conts) else "(no data)"
                model_entries.append((name, text, 0))
            else:
                full_texts = mdata.get("full_texts", [])
                boundaries = mdata.get("prompt_boundaries", [])
                ft = full_texts[idx] if idx < len(full_texts) else "(no data)"
                boundary = boundaries[idx] if idx < len(boundaries) else 0
                model_entries.append((name, ft, boundary))

        output = format_example(
            idx, n_examples, prompt_text, full_original, prompt_len,
            model_entries, args.wrap,
        )
        print(output)

        if args.all:
            idx += 1
            continue

        try:
            choice = input(f"\n[{idx+1}/{n_examples}] > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if choice in ("q", "quit", "exit"):
            break
        elif choice in ("p", "prev"):
            idx = max(0, idx - 1)
        elif choice.isdigit():
            target = int(choice) - 1
            idx = max(0, min(target, n_examples - 1))
        else:
            idx += 1


if __name__ == "__main__":
    main()
