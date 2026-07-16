"""Configuration for the data_loader stage."""

from dataclasses import dataclass
from typing import Optional

from gen_gec_errant._config_utils import load_config_from_yaml as _load, apply_cli_overrides as _apply

_SECTION_MAP: dict = {}


@dataclass
class DataLoaderConfig:
    """Configuration for loading and preparing input data."""
    data_path: str = "data/efcamdat_sentences.txt"
    max_sentences: Optional[int] = None
    min_words: int = 10
    max_words: int = 10000
    prompt_ratio: float = 0.5
    min_prompt_words: int = 3
    text_column: Optional[str] = None
    split_sentences: bool = True


def load_config_from_yaml(path: str) -> DataLoaderConfig:
    return _load(path, DataLoaderConfig, _SECTION_MAP)


def apply_cli_overrides(config: DataLoaderConfig, overrides: list[str]) -> DataLoaderConfig:
    return _apply(config, overrides, _SECTION_MAP)
