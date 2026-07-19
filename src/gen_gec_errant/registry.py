"""Central registry for models, datasets, paths, and pipeline config building."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from gen_gec_errant.generation.config import ModelConfig
from gen_gec_errant.pipeline.config import PipelineConfig


# ── Dataset config ─────────────────────────────────────────────────────

@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""
    name: str
    filename: str
    description: str
    text_column: str = "text"


# ── Path config ────────────────────────────────────────────────────────

@dataclass
class PathConfig:
    """Root paths for data, models, and output."""
    data_root: Path
    models_root: Path
    output_root: Path

    @classmethod
    def for_colab(cls) -> "PathConfig":
        gdrive = Path("/content/drive/MyDrive")
        return cls(
            data_root=gdrive / "phd-experimental-data/cefr-classification/data/splits",
            models_root=gdrive / "_p/artificial-learners/models",
            output_root=gdrive / "_p/artificial-learners/generation-gec-errant/results",
        )

    @classmethod
    def for_local(cls) -> "PathConfig":
        return cls(
            data_root=Path("data/splits"),
            models_root=Path("models"),
            output_root=Path("results"),
        )

    def dataset_path(self, dataset: DatasetConfig) -> Path:
        return self.data_root / dataset.filename

    def model_gdrive_path(self, model: ModelConfig) -> Optional[Path]:
        if model.gdrive_subpath is None:
            return None
        return self.models_root / model.gdrive_subpath / model.checkpoint_subdir


# ── Model registry ─────────────────────────────────────────────────────

MODEL_REGISTRY: Dict[str, ModelConfig] = {
    # ── Native (pre-trained) baselines ──
    "gpt2-small-native": ModelConfig(
        name="gpt2-small-native",
        params="124M",
        hf_model_id="gpt2",
        model_family="gpt2",
        gdrive_subpath=None,
        is_learner_tuned=False,
        batch_size=16,
        gec_batch_size=32,
        description="GPT-2 Small native (zero-shot baseline)",
    ),
    "gpt2-medium-native": ModelConfig(
        name="gpt2-medium-native",
        params="355M",
        hf_model_id="gpt2-medium",
        model_family="gpt2",
        gdrive_subpath=None,
        is_learner_tuned=False,
        batch_size=8,
        gec_batch_size=32,
        description="GPT-2 Medium native (zero-shot baseline; matched control for ft-gpt2-medium)",
    ),
    "gpt2-large-native": ModelConfig(
        name="gpt2-large-native",
        params="774M",
        hf_model_id="gpt2-large",
        model_family="gpt2",
        gdrive_subpath=None,
        is_learner_tuned=False,
        batch_size=4,
        gec_batch_size=16,
        description="GPT-2 Large native (zero-shot baseline; matched control for ft-gpt2-large)",
    ),
    # ── GPT-2 fine-tuned ──
    "ft-gpt2-small": ModelConfig(
        name="ft-gpt2-small",
        params="124M",
        model_family="gpt2",
        gdrive_subpath="gpt2/gpt2-small-all-data",
        checkpoint_subdir="best/checkpoint-7596",
        is_learner_tuned=True,
        batch_size=16,
        gec_batch_size=32,
        description="GPT-2 Small (124M) fine-tuned on EFCAMDAT all-data",
    ),
    "ft-gpt2-medium": ModelConfig(
        name="ft-gpt2-medium",
        params="355M",
        model_family="gpt2",
        gdrive_subpath="gpt2/gpt2-medium-all-data",
        checkpoint_subdir="best/checkpoint-5625",
        is_learner_tuned=True,
        batch_size=8,
        gec_batch_size=32,
        description="GPT-2 Medium (355M) fine-tuned on EFCAMDAT all-data",
    ),
    "ft-gpt2-large": ModelConfig(
        name="ft-gpt2-large",
        params="774M",
        model_family="gpt2",
        gdrive_subpath="gpt2/gpt2-large-all-data",
        checkpoint_subdir="best/checkpoint-6750",
        is_learner_tuned=True,
        batch_size=4,
        gec_batch_size=16,
        description="GPT-2 Large (774M) fine-tuned on EFCAMDAT all-data",
    ),
    # ── Pythia fine-tuned ──
    "ft-pythia-70m": ModelConfig(
        name="ft-pythia-70m",
        params="70M",
        model_family="pythia",
        gdrive_subpath="pythia/pythia-70m-all-data",
        is_learner_tuned=True,
        batch_size=16,
        gec_batch_size=32,
        description="Pythia 70M fine-tuned on EFCAMDAT all-data",
    ),
    "ft-pythia-160m": ModelConfig(
        name="ft-pythia-160m",
        params="160M",
        model_family="pythia",
        gdrive_subpath="pythia/pythia-160m-all-data",
        is_learner_tuned=True,
        batch_size=16,
        gec_batch_size=32,
        description="Pythia 160M fine-tuned on EFCAMDAT all-data",
    ),
    "ft-pythia-410m": ModelConfig(
        name="ft-pythia-410m",
        params="410M",
        model_family="pythia",
        gdrive_subpath="pythia/pythia-410m-all-data",
        checkpoint_subdir="best/checkpoint-21476",
        is_learner_tuned=True,
        batch_size=8,
        gec_batch_size=32,
        description="Pythia 410M fine-tuned on EFCAMDAT all-data",
    ),
    "ft-pythia-1b": ModelConfig(
        name="ft-pythia-1b",
        params="1B",
        model_family="pythia",
        gdrive_subpath="pythia/pythia-1b-all-data",
        checkpoint_subdir="best/checkpoint-21036",
        is_learner_tuned=True,
        batch_size=4,
        gec_batch_size=16,
        description="Pythia 1B fine-tuned on EFCAMDAT all-data",
    ),
    "ft-pythia-1.4b": ModelConfig(
        name="ft-pythia-1.4b",
        params="1.4B",
        model_family="pythia",
        gdrive_subpath="pythia/pythia-1.4b-all-data",
        is_learner_tuned=True,
        batch_size=2,
        gec_batch_size=16,
        description="Pythia 1.4B fine-tuned on EFCAMDAT all-data",
    ),
    "pythia-410m-native": ModelConfig(
        name="pythia-410m-native",
        params="410M",
        hf_model_id="EleutherAI/pythia-410m",
        model_family="pythia",
        gdrive_subpath=None,
        is_learner_tuned=False,
        batch_size=8,
        gec_batch_size=32,
        description="Pythia 410M native (zero-shot baseline; matched control for ft-pythia-410m)",
    ),
    # ── SmolLM2 fine-tuned ──
    "ft-smollm2-135m": ModelConfig(
        name="ft-smollm2-135m",
        params="135M",
        model_family="smollm2",
        gdrive_subpath="smollm2/smollm2-135m-all-data",
        is_learner_tuned=True,
        batch_size=16,
        gec_batch_size=32,
        description="SmolLM2 135M fine-tuned on EFCAMDAT all-data",
    ),
    "ft-smollm2-360m": ModelConfig(
        name="ft-smollm2-360m",
        params="360M",
        model_family="smollm2",
        gdrive_subpath="smollm2/smollm2-360m-all-data",
        checkpoint_subdir="best/checkpoint-16712",
        is_learner_tuned=True,
        batch_size=8,
        gec_batch_size=32,
        description="SmolLM2 360M fine-tuned on EFCAMDAT all-data",
    ),
    "ft-smollm2-1.7b": ModelConfig(
        name="ft-smollm2-1.7b",
        params="1.7B",
        model_family="smollm2",
        gdrive_subpath="smollm2/smollm2-1.7b-all-data",
        is_learner_tuned=True,
        batch_size=2,
        gec_batch_size=16,
        description="SmolLM2 1.7B fine-tuned on EFCAMDAT all-data",
    ),
    "smollm2-360m-native": ModelConfig(
        name="smollm2-360m-native",
        params="360M",
        hf_model_id="HuggingFaceTB/SmolLM2-360M",
        model_family="smollm2",
        gdrive_subpath=None,
        is_learner_tuned=False,
        batch_size=8,
        gec_batch_size=32,
        description="SmolLM2 360M native (zero-shot baseline; matched control for ft-smollm2-360m)",
    ),
}

# ── Dataset registry ───────────────────────────────────────────────────

DATASET_REGISTRY: Dict[str, DatasetConfig] = {
    "norm-CELVA-SP": DatasetConfig(
        name="norm-CELVA-SP",
        filename="norm-CELVA-SP.csv",
        description="Spanish L1 learner English (primary)",
    ),
    "norm-EFCAMDAT-test": DatasetConfig(
        name="norm-EFCAMDAT-test",
        filename="norm-EFCAMDAT-test.csv",
        description="In-domain learner English",
    ),
    "norm-KUPA-KEYS": DatasetConfig(
        name="norm-KUPA-KEYS",
        filename="norm-KUPA-KEYS.csv",
        description="Cross-corpus learner English",
    ),
}

# ── Pipeline defaults ──────────────────────────────────────────────────

PIPELINE_DEFAULTS = {
    "generation": {
        "max_new_tokens": 50,
        "min_new_tokens": 10,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 0.95,
        "do_sample": True,
        "repetition_penalty": 1.2,
    },
    "data_loader": {
        "text_column": "text",
        "min_words": 10,
        "max_words": 500,
        "prompt_ratio": 0.5,
        "min_prompt_words": 5,
    },
    "gec": {
        "method": "dedicated",
        "model_id": "grammarly/coedit-large",
        "device": "auto",
    },
    "annotation": {"lang": "en"},
    "analysis": {"skip_plots": False, "top_n_error_types": 10},
    "seed": 42,
}


# ── Accessor helpers ───────────────────────────────────────────────────

def get_models(keys: Optional[List[str]] = None) -> List[ModelConfig]:
    """Return model configs. All models if keys is None."""
    if keys is None:
        return list(MODEL_REGISTRY.values())
    return [MODEL_REGISTRY[k] for k in keys]


def get_datasets(keys: Optional[List[str]] = None) -> List[DatasetConfig]:
    """Return dataset configs. All datasets if keys is None."""
    if keys is None:
        return list(DATASET_REGISTRY.values())
    return [DATASET_REGISTRY[k] for k in keys]


# ── Pipeline config builder ───────────────────────────────────────────

def build_pipeline_config(
    model: ModelConfig,
    dataset: DatasetConfig,
    paths: PathConfig,
    *,
    model_path: Optional[str] = None,
    max_sentences: Optional[int] = None,
    include_learner_baseline: bool = True,
    output_dir: Optional[str] = None,
) -> PipelineConfig:
    """Build a PipelineConfig from registry entries.

    Args:
        model: ModelConfig from the registry.
        dataset: DatasetConfig from the registry.
        paths: PathConfig with root directories.
        model_path: Resolved model path (local SSD or HF id).
            If None, uses hf_model_id for native models or the gdrive path.
        max_sentences: Limit on number of sentences (None = all).
        include_learner_baseline: Whether to add learner baseline.
        output_dir: Override output directory.
    """
    # Resolve model path
    if model_path is None:
        if model.gdrive_subpath is not None:
            gdrive_path = paths.model_gdrive_path(model)
            model_path = str(gdrive_path) if gdrive_path else model.hf_model_id
        else:
            model_path = model.hf_model_id

    defaults = PIPELINE_DEFAULTS

    from gen_gec_errant.data_loader.config import DataLoaderConfig
    from gen_gec_errant.generation.config import GenerationParams
    from gen_gec_errant.gec.config import GECConfig
    from gen_gec_errant.annotation.config import AnnotationConfig
    from gen_gec_errant.analysis.config import AnalysisConfig

    data_path = str(paths.dataset_path(dataset))

    return PipelineConfig(
        data_loader=DataLoaderConfig(
            data_path=data_path,
            text_column=dataset.text_column,
            max_sentences=max_sentences,
            min_words=defaults["data_loader"]["min_words"],
            max_words=defaults["data_loader"]["max_words"],
            prompt_ratio=defaults["data_loader"]["prompt_ratio"],
            min_prompt_words=defaults["data_loader"]["min_prompt_words"],
        ),
        generation=GenerationParams(**defaults["generation"]),
        gec=GECConfig(
            method=defaults["gec"]["method"],
            model_id=defaults["gec"]["model_id"],
            batch_size=model.gec_batch_size,
            device=defaults["gec"]["device"],
        ),
        annotation=AnnotationConfig(**defaults["annotation"]),
        analysis=AnalysisConfig(**defaults["analysis"]),
        models=[{
            "name": model.name,
            "hf_model_id": model_path,
            "model_family": model.model_family,
            "is_learner_tuned": model.is_learner_tuned,
        }],
        batch_size=model.batch_size,
        device="auto",
        seed=defaults["seed"],
        output_dir=output_dir or "results",
        include_learner_baseline=include_learner_baseline,
    )
