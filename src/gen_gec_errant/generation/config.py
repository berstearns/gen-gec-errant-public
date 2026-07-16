"""Configuration for the generation stage."""

from dataclasses import dataclass, field
from typing import Optional

from gen_gec_errant._config_utils import load_config_from_yaml as _load, apply_cli_overrides as _apply


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str = ""
    hf_model_id: str = "gpt2"
    checkpoint_path: Optional[str] = None
    model_family: str = "gpt2"
    is_learner_tuned: bool = False
    # Registry fields (optional — ignored by YAML configs that don't set them)
    params: Optional[str] = None
    description: Optional[str] = None
    gdrive_subpath: Optional[str] = None
    checkpoint_subdir: str = "final"
    batch_size: int = 8
    gec_batch_size: int = 32


@dataclass
class GenerationParams:
    """Parameters for text generation."""
    max_new_tokens: int = 50
    min_new_tokens: int = 10
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    do_sample: bool = True
    num_return_sequences: int = 1
    repetition_penalty: float = 1.2


@dataclass
class GenerationConfig:
    """Full configuration for the generation stage."""
    params: GenerationParams = field(default_factory=GenerationParams)
    model: ModelConfig = field(default_factory=ModelConfig)
    batch_size: int = 8
    device: str = "auto"
    seed: int = 42


_SECTION_MAP = {
    "params": GenerationParams,
    "model": ModelConfig,
}


def load_config_from_yaml(path: str) -> GenerationConfig:
    return _load(path, GenerationConfig, _SECTION_MAP)


def apply_cli_overrides(config: GenerationConfig, overrides: list[str]) -> GenerationConfig:
    return _apply(config, overrides, _SECTION_MAP)
