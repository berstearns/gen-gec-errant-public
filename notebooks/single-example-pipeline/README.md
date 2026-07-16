# Single-Example Pipeline — step-by-step marimo notebook

One authentic-learner sentence, run through the **full pipeline**
(generation → GEC → ERRANT → compare), with **every stage's config + input + output**
shown, ending in a **three-way comparison**: authentic learner vs artificial learner (AL)
vs matched control. Built for debugging / complete ownership of each step — **n=1, a
pipeline debugger, not evidence** (see `SPEC.md` §8, `gen-gec-review-specs/`).

## Contents

| Path | What it is |
|------|-----------|
| `single_example_pipeline.py` | **The marimo notebook** (the deliverable) |
| `single_example_pipeline.ipynb` | Colab export (fallback for Google Colab) |
| `SPEC.md` | The build spec — what the notebook must do + the acceptance gate |
| `REPORT-to-reviewer.md` | The executor's report on the build/run |
| `gate/` | The verification-run artifacts (see below) |

### `gate/` — verification run
- `trace_single_example.py`, `run_gate*.sh` — the scripts that exercised the pipeline
- `00-gate*.log`, `02-fresh-import.log`, `03*-export.log`, `04-post-install.log`,
  `05-trace.log`, `06-vocab.log`, `07*-ipynb.log`, `08-ipynb-exec.log`, `gate*-run.log`
  — install / import / export / trace / vocab gate logs
- `executed_single_example.ipynb` — the notebook **executed, with outputs**
- `export.html` — rendered notebook (open in any browser)
- `compare-chart.png` — the 3-source generation-region error-tag comparison chart

## Run it

**Local (marimo):** launch from the **repo root** so the `models/` symlink resolves the
AL checkpoint:
```bash
marimo edit notebooks/single-example-pipeline/single_example_pipeline.py
```

**Colab:** open `single_example_pipeline.ipynb` (Cell 0 self-installs).

## The three sources compared (CONCEPTS.md vocabulary)

- **authentic learner** — the sentence's real human continuation (the yardstick).
- **artificial learner (AL)** — `ft-gpt2-small`; checkpoint at
  `models/gpt2/gpt2-small-all-data/best/checkpoint-7596` (repo `models/` → symlink to
  `PLACEHOLDER_MODELS_ROOT`).
- **matched control** — `gpt2` (native, same architecture + size).

Objective: **distance of the AL error-tag distribution to the authentic learner**, vs the
matched control — never bare "AL errs more."
