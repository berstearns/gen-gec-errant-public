#!/usr/bin/env python3
"""Generate the 10 S1 core-contrast configs (5 matched pairs) from the registry.

5 matched pairs per 20-preregistration.md / 60-results-ledger.md S1 rows:
gpt2 small/medium/large, pythia 410m/1b — pretrained vs the exact registry
fine-tuned best/ checkpoint. CELVA-SP full (no max_sentences cap), coedit-large,
seed 42. The reference-annotation leg (include_learner_baseline) runs in only
ONE of the 10 configs (CANONICAL_LEARNER_BASELINE_LABEL) — it reprofiles the
same reference continuations regardless of generator, so leaving it on
everywhere wasted ~30% of S1 GPU spend on 10 identical copies (dispatch #3).

Prep only — see gen-gec-review-specs/50-dispatch-log.md dispatch #2/#3. This
script does not run anything; it only writes config files.

Usage:
    python scripts/gen_s1_configs.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from gen_gec_errant.registry import MODEL_REGISTRY, PathConfig  # noqa: E402

LOCAL_MODELS_ROOT = Path("PLACEHOLDER_MODELS_ROOT")
DATA_PATH = REPO_ROOT / "data" / "norm-CELVA-SP.csv"
OUT_DIR = REPO_ROOT / "configs" / "s1-core-contrast"

# Dispatch #3 (50-dispatch-log.md, finding #2): the learner-baseline GEC leg
# reprofiles the SAME reference continuations regardless of which generator is
# under test, so running it in all 10 configs wasted ~30% of S1 GPU spend on
# 10 identical copies. Only one config keeps it on — its learner_profile.json
# is the canonical reference profile every S1 cell compares against.
CANONICAL_LEARNER_BASELINE_LABEL = "gpt2-small"

# (pretrained_name, pretrained_hf_id, ft_registry_key, family)
PAIRS = [
    ("gpt2-small", "gpt2", "ft-gpt2-small", "gpt2"),
    ("gpt2-medium", "gpt2-medium", "ft-gpt2-medium", "gpt2"),
    ("gpt2-large", "gpt2-large", "ft-gpt2-large", "gpt2"),
    ("pythia-410m", "EleutherAI/pythia-410m", "ft-pythia-410m", "pythia"),
    ("pythia-1b", "EleutherAI/pythia-1b", "ft-pythia-1b", "pythia"),
]

TEMPLATE = """\
# S1 core contrast — {label} ({kind}) — CELVA-SP full
# gen-gec-review-specs/50-dispatch-log.md dispatch #2 (S1 PREP). DO NOT LAUNCH
# without explicit go-ahead — GPU spend is the user's call (90-open-questions.md #3).
# Usage: python -m gen_gec_errant.pipeline --config configs/s1-core-contrast/{label}.yaml
data_loader:
  data_path: {data_path}
  text_column: text
  max_sentences: null
  min_words: 10
  max_words: 500
  prompt_ratio: 0.5
  min_prompt_words: 5
  split_sentences: true

generation:
  max_new_tokens: 50
  min_new_tokens: 10
  temperature: 1.0
  top_k: 50
  top_p: 0.95
  do_sample: true
  repetition_penalty: 1.2

gec:
  method: dedicated
  model_id: grammarly/coedit-large
  batch_size: {gec_batch_size}
  device: auto

annotation:
  lang: en

analysis:
  skip_plots: false
  top_n_error_types: 10

models:
  - name: {label}
    hf_model_id: {hf_id}
    model_family: {family}
    {is_learner_tuned_line}batch_size: {gen_batch_size}

batch_size: {gen_batch_size}
device: auto
seed: 42
output_dir: outputs/s1-core-contrast/{label}
skip_plots: false
{learner_baseline_comment}include_learner_baseline: {learner_baseline}
"""


def write_config(label, hf_id, family, is_ft, gen_batch_size, gec_batch_size):
    kind = "fine-tuned" if is_ft else "pretrained"
    is_lt = "is_learner_tuned: true\n    " if is_ft else ""
    is_canonical = label == CANONICAL_LEARNER_BASELINE_LABEL
    if is_canonical:
        comment = (
            "# Canonical learner_profile.json for S1 — every other cell reuses this\n"
            "# one instead of re-running the same reference GEC leg (dispatch #3).\n"
        )
    else:
        comment = (
            f"# off: {CANONICAL_LEARNER_BASELINE_LABEL}.yaml already produced the reference\n"
            "# learner_profile.json; re-running it here would waste GPU on an identical\n"
            "# GEC pass over the same reference continuations (dispatch #3).\n"
        )
    text = TEMPLATE.format(
        label=label, kind=kind, data_path=DATA_PATH, hf_id=hf_id, family=family,
        is_learner_tuned_line=is_lt,
        gen_batch_size=gen_batch_size, gec_batch_size=gec_batch_size,
        learner_baseline_comment=comment,
        learner_baseline="true" if is_canonical else "false",
    )
    out_path = OUT_DIR / f"{label}.yaml"
    out_path.write_text(text)
    return out_path


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = PathConfig(
        data_root=REPO_ROOT / "data",
        models_root=LOCAL_MODELS_ROOT,
        output_root=REPO_ROOT / "outputs",
    )

    written = []
    for pretrained_name, pretrained_hf_id, ft_key, family in PAIRS:
        ft_model = MODEL_REGISTRY[ft_key]
        ft_local_path = paths.model_gdrive_path(ft_model)

        written.append(write_config(
            pretrained_name, pretrained_hf_id, family, is_ft=False,
            gen_batch_size=ft_model.batch_size, gec_batch_size=ft_model.gec_batch_size,
        ))
        written.append(write_config(
            ft_key, str(ft_local_path), family, is_ft=True,
            gen_batch_size=ft_model.batch_size, gec_batch_size=ft_model.gec_batch_size,
        ))

    print(f"Wrote {len(written)} configs to {OUT_DIR}")
    for p in written:
        print(" ", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
