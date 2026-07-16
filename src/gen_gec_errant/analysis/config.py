"""Configuration for the analysis stage."""

from dataclasses import dataclass

from gen_gec_errant._config_utils import load_config_from_yaml as _load, apply_cli_overrides as _apply

_SECTION_MAP: dict = {}


@dataclass
class AnalysisConfig:
    """Configuration for analysis and visualization."""
    output_dir: str = "results"
    skip_plots: bool = False
    top_n_error_types: int = 10


def load_config_from_yaml(path: str) -> AnalysisConfig:
    return _load(path, AnalysisConfig, _SECTION_MAP)


def apply_cli_overrides(config: AnalysisConfig, overrides: list[str]) -> AnalysisConfig:
    return _apply(config, overrides, _SECTION_MAP)
