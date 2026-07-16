"""Configuration for the GEC stage."""

from dataclasses import dataclass
from typing import Optional

from gen_gec_errant._config_utils import load_config_from_yaml as _load, apply_cli_overrides as _apply

_SECTION_MAP: dict = {}


@dataclass
class GECConfig:
    """Configuration for grammatical error correction."""
    method: str = "dedicated"  # "llm" or "dedicated"
    model_id: str = "grammarly/coedit-large"
    prompt_template: str = (
        "Correct any grammatical errors in the following sentence. "
        "Only fix grammar — do not change meaning, vocabulary, or style. "
        "If the sentence is already correct, return it unchanged.\n\n"
        "Sentence: {sentence}\n\n"
        "Corrected sentence:"
    )
    batch_size: int = 32
    device: str = "auto"


def load_config_from_yaml(path: str) -> GECConfig:
    return _load(path, GECConfig, _SECTION_MAP)


def apply_cli_overrides(config: GECConfig, overrides: list[str]) -> GECConfig:
    return _apply(config, overrides, _SECTION_MAP)
