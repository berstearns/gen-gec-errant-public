"""gen-gec-errant: automatic text generation, GEC, and ERRANT error annotation."""

__version__ = "0.1.0"


def __getattr__(name):
    """Lazy imports for convenience — avoids loading torch/errant at import time."""
    _imports = {
        "run_data_loader": "gen_gec_errant.data_loader",
        "run_generation": "gen_gec_errant.generation",
        "run_gec": "gen_gec_errant.gec",
        "run_annotation": "gen_gec_errant.annotation",
        "run_analysis": "gen_gec_errant.analysis",
        "run_pipeline": "gen_gec_errant.pipeline",
    }
    if name in _imports:
        import importlib
        mod = importlib.import_module(_imports[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'gen_gec_errant' has no attribute {name!r}")
