"""Configuration for the annotation stage."""

from dataclasses import dataclass

from gen_gec_errant._config_utils import load_config_from_yaml as _load, apply_cli_overrides as _apply

_SECTION_MAP: dict = {}


@dataclass
class AnnotationConfig:
    """Configuration for ERRANT error annotation."""
    lang: str = "en"


def load_config_from_yaml(path: str) -> AnnotationConfig:
    return _load(path, AnnotationConfig, _SECTION_MAP)


def apply_cli_overrides(config: AnnotationConfig, overrides: list[str]) -> AnnotationConfig:
    return _apply(config, overrides, _SECTION_MAP)
