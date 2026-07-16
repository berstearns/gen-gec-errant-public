"""Configuration for the preprocessing stage."""

from dataclasses import dataclass
from typing import Optional

from gen_gec_errant._config_utils import load_config_from_yaml as _load, apply_cli_overrides as _apply

_SECTION_MAP: dict = {}


@dataclass
class PreprocessingConfig:
    """Configuration for EFCAMDAT preprocessing."""
    input_path: str = "data/cleaned_efcamdat.csv"
    output_path: str = "data/efcamdat_sentences.csv"
    min_words: int = 5
    max_words: int = 60
    max_essays: Optional[int] = None
    cefr_filter: Optional[str] = None   # e.g. "A1,A2,B1"
    l1_filter: Optional[str] = None     # e.g. "Arabic,Mandarin"
    text_col: Optional[str] = None
    corrected_col: Optional[str] = None


def load_config_from_yaml(path: str) -> PreprocessingConfig:
    return _load(path, PreprocessingConfig, _SECTION_MAP)


def apply_cli_overrides(config: PreprocessingConfig, overrides: list[str]) -> PreprocessingConfig:
    return _apply(config, overrides, _SECTION_MAP)
