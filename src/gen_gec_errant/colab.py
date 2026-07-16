"""Colab-specific helpers for model resolution and cleanup."""

import shutil
from pathlib import Path
from typing import Optional

from gen_gec_errant.generation.config import ModelConfig
from gen_gec_errant.registry import PathConfig


def is_colab() -> bool:
    """Detect if running inside Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def resolve_model_path(
    model: ModelConfig,
    paths: PathConfig,
    local_cache_root: Path = Path("/content/models"),
) -> Optional[str]:
    """Resolve a model to a loadable path, copying from GDrive to local SSD if needed.

    Returns:
        Local path string, HuggingFace model ID, or None if not found.
    """
    if model.gdrive_subpath is None:
        return model.hf_model_id

    gdrive_path = paths.model_gdrive_path(model)
    if gdrive_path is None or not gdrive_path.exists():
        print(f"  WARNING: Model not found: {gdrive_path}")
        return None

    local_cache = local_cache_root / model.name
    if not local_cache.exists():
        print(f"  Copying to local SSD...")
        shutil.copytree(gdrive_path, local_cache)
        size_gb = sum(f.stat().st_size for f in local_cache.rglob("*") if f.is_file()) / 1e9
        print(f"  Done. Size: {size_gb:.2f} GB")
    else:
        print(f"  Model already cached locally: {local_cache}")

    return str(local_cache)


def cleanup_local_model(
    model: ModelConfig,
    local_cache_root: Path = Path("/content/models"),
) -> None:
    """Remove local model cache to free disk space."""
    local_cache = local_cache_root / model.name
    if local_cache.exists():
        shutil.rmtree(local_cache)
        print(f"  Cleaned up local cache: {local_cache}")
