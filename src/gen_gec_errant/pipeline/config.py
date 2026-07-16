"""Configuration for the full pipeline."""

from dataclasses import dataclass, field
from typing import List, Optional

from gen_gec_errant._config_utils import (
    load_config_from_yaml as _load,
    apply_cli_overrides as _apply,
    build_sub_config,
)
from gen_gec_errant.data_loader.config import DataLoaderConfig
from gen_gec_errant.generation.config import GenerationConfig, GenerationParams, ModelConfig
from gen_gec_errant.gec.config import GECConfig
from gen_gec_errant.annotation.config import AnnotationConfig
from gen_gec_errant.analysis.config import AnalysisConfig


@dataclass
class PipelineConfig:
    """Master configuration for the full pipeline."""
    data_loader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    generation: GenerationParams = field(default_factory=GenerationParams)
    gec: GECConfig = field(default_factory=GECConfig)
    annotation: AnnotationConfig = field(default_factory=AnnotationConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    # Models to evaluate
    models: List[dict] = field(default_factory=lambda: [
        {"name": "gpt2-base", "hf_model_id": "gpt2", "model_family": "gpt2"},
    ])

    # Global settings
    batch_size: int = 8
    device: str = "auto"
    seed: int = 42
    output_dir: str = "results"

    # Skip flags
    skip_generation: bool = False
    skip_gec: bool = False
    skip_plots: bool = False

    # G3 reference-annotation leg: run the actual learner's reference
    # continuation through the same GEC+ERRANT instrument as every model
    # (as a pseudo-model "learner_baseline"), producing learner_profile.json.
    include_learner_baseline: bool = True

    # Resume from last completed checkpoint
    resume: bool = False


_SECTION_MAP = {
    "data_loader": DataLoaderConfig,
    "generation": GenerationParams,
    "gec": GECConfig,
    "annotation": AnnotationConfig,
    "analysis": AnalysisConfig,
}


def load_config_from_yaml(path: str) -> PipelineConfig:
    return _load(path, PipelineConfig, _SECTION_MAP)


def apply_cli_overrides(config: PipelineConfig, overrides: list[str]) -> PipelineConfig:
    return _apply(config, overrides, _SECTION_MAP)


def get_model_configs(config: PipelineConfig) -> List[ModelConfig]:
    """Convert the models list-of-dicts to ModelConfig objects."""
    return [build_sub_config(ModelConfig, m) for m in config.models]
