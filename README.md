# gen-gec-errant

Pipeline for automatic text generation, grammatical error correction (GEC), and ERRANT error annotation.

## Installation

```bash
pip install -e .
python -m spacy download en_core_web_sm
```

## Quick Start

### Full pipeline (CLI)

```bash
python -m gen_gec_errant.pipeline --config configs/pipeline/quickstart.yaml
```

### Single stage (CLI)

```bash
python -m gen_gec_errant.generation --config configs/generation/default.yaml --data_path data/sentences.txt
```

### CLI overrides

```bash
python -m gen_gec_errant.pipeline --config configs/pipeline/default.yaml \
    --generation.temperature 0.7 \
    --batch_size 16
```

### Python / Jupyter

```python
from gen_gec_errant.data_loader import run_data_loader
from gen_gec_errant.data_loader.config import load_config_from_yaml

config = load_config_from_yaml("configs/data_loader/default.yaml")
items = run_data_loader(config)
```

## Pipeline Stages

| Stage | Module | Runner |
|-------|--------|--------|
| 1. Data Loading | `gen_gec_errant.data_loader` | `run_data_loader(config)` |
| 2. Generation | `gen_gec_errant.generation` | `run_generation(config, items)` |
| 3. GEC | `gen_gec_errant.gec` | `run_gec(config, results)` |
| 4. Annotation | `gen_gec_errant.annotation` | `run_annotation(config, results)` |
| 5. Analysis | `gen_gec_errant.analysis` | `run_analysis(config, all_results, items)` |
| Orchestrator | `gen_gec_errant.pipeline` | `run_pipeline(config)` |
| Preprocessing | `gen_gec_errant.preprocessing` | `run_preprocessing(config)` |

## Project Structure

```
src/gen_gec_errant/
    __init__.py              # __version__, convenience imports
    _config_utils.py         # YAML loading, CLI overrides
    _types.py                # ErrorAnnotation, SentenceAnnotation
    data_loader/             # Stage 1: load sentences, make prompts
    generation/              # Stage 2: generate text, compute perplexity
    gec/                     # Stage 3: grammatical error correction
    annotation/              # Stage 4: ERRANT error annotation
    analysis/                # Stage 5: statistics, plots, CSV export
    pipeline/                # Orchestrator
    preprocessing/           # EFCAMDAT preprocessing
configs/                     # YAML configuration files
notebooks/                   # Jupyter notebooks
```

## Configuration

Each stage has its own YAML config under `configs/`. The pipeline config nests all stages:

```yaml
data_loader:
  data_path: data/sentences.txt
  max_sentences: 100

generation:
  temperature: 1.0
  max_new_tokens: 50

gec:
  method: dedicated
  model_id: grammarly/coedit-large

models:
  - name: gpt2-base
    hf_model_id: gpt2
```

## Available Models

### GEC models

The `gec` config takes a `method` and a `model_id`. Each config specifies only the model it uses.

| Method | Model ID | Description |
|--------|----------|-------------|
| `dedicated` | `grammarly/coedit-large` | Purpose-built seq2seq GEC model (default) |
| `llm` | `google/gemma-2-2b-it` | Instruction-tuned LLM with prompt-based correction |

### Generation models

Any HuggingFace causal LM works. Tested models:

| Name | HF Model ID |
|------|-------------|
| `gpt2-base` | `gpt2` |
| `gpt2-medium` | `gpt2-medium` |
| `gpt2-large` | `gpt2-large` |
| `pythia-70m` | `EleutherAI/pythia-70m` |
| `pythia-160m` | `EleutherAI/pythia-160m` |
| `pythia-410m` | `EleutherAI/pythia-410m` |
| `pythia-1b` | `EleutherAI/pythia-1b` |

Fine-tuned checkpoints can be loaded by pointing `hf_model_id` to a local path, or by setting `is_learner_tuned: true` with a `checkpoint_path` to load weights on top of a base model.
