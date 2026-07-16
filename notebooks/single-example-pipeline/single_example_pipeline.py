"""Single-example, step-by-step trace of the gen -> GEC -> ERRANT -> compare pipeline.

This is a *marimo* notebook (pure Python, reactive). It runs the ENTIRE production
instrument (`src/gen_gec_errant/`) on ONE hardcoded learner sentence at a time,
printing every stage's config + input + output, and ends in a detailed comparison of
the three error-producing sources on that one sentence:

    authentic learner  (the YARDSTICK)  ·  matched control  ·  artificial learner (AL)

Vocabulary is canonical per CONCEPTS.md. The scientific quantity is *distance to the
authentic learner* — never "the AL errs more". This notebook is a single-example
DEBUGGER for human ownership, NOT evidence: the real, pre-registered, distributional
measurement lives in the batch pipeline (see the n=1 caveat at the very end).

Launch locally:   marimo edit notebooks/single_example_pipeline.py
Headless run:     marimo export html notebooks/single_example_pipeline.py -o /tmp/out.html
Colab fallback:   notebooks/single_example_pipeline.ipynb  (marimo export ipynb)

Verified end-to-end on **Python 3.13.11 (CPU)**; the exact known-good package set is frozen in
`notebooks/requirements-lock-py3.13.txt` (see the reproducibility cell). Those are the exact
versions that ran, not `>=` ranges.
"""

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


app._unparsable_cell(
    r"""
    wimport marimo as mo
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Single-example pipeline trace — gen → GEC → ERRANT → compare

    One learner sentence, every stage laid bare, ending in a three-source comparison
    against the **authentic learner** (the yardstick).

    > **Sources (CONCEPTS.md canonical):**
    > **authentic learner** — the real human L2 continuation, the yardstick ·
    > **matched control** — the native-pretrained model (`gpt2`) ·
    > **artificial learner (AL)** — the same architecture *fine-tuned on learner text*
    > (`ft-gpt2-small`). The matched pair `ft-gpt2-small : gpt2` differs *only* by fine-tuning.
    >
    > The quantity of interest is **distance to the authentic learner's error-tag
    > distribution**, concentrated in acquisition categories — *not* raw error counts.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Cell 0 — Environment + install (idempotent)
    """)
    return


@app.cell
def _():
    # -- Cell 0: make a fresh runtime (local venv OR a clean Colab) work with no manual pip.
    # Everything shells out via subprocess (no !pip / no magics — this is marimo, not Jupyter).
    import importlib
    import os
    import subprocess
    import sys
    from pathlib import Path

    def _colab() -> bool:
        try:
            import google.colab  # noqa: F401

            return True
        except Exception:
            return False

    IN_COLAB = _colab()

    def _pip(*args):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

    def _find_repo_root():
        here = Path(os.getcwd()).resolve()
        for cand in [here, *here.parents]:
            if (cand / "pyproject.toml").exists() and (cand / "src" / "gen_gec_errant").exists():
                return cand
        return None

    _log = []

    def _ensure(import_name, pip_args, label=None):
        label = label or import_name
        try:
            importlib.import_module(import_name)
            _log.append(f"✓ {label}: already importable")
            return
        except Exception:
            _log.append(f"… {label}: installing (pip install {' '.join(pip_args)})")
            _pip(*pip_args)
            importlib.invalidate_caches()
            _log.append(f"✓ {label}: installed")

    # marimo itself (belt-and-suspenders for the exported .ipynb Colab path).
    _ensure("marimo", ["marimo"])

    # The pipeline package. Editable install from the repo locally; git install on a bare Colab.
    _repo_root = _find_repo_root()
    try:
        importlib.import_module("gen_gec_errant")
        _log.append("✓ gen_gec_errant: already importable")
    except Exception:
        if _repo_root is not None:
            _log.append(f"… gen_gec_errant: pip install -e {_repo_root}")
            _pip("-e", str(_repo_root))
            # An editable install writes src/ into a .pth file that Python reads only at
            # interpreter *startup*; a pip install inside THIS running process therefore
            # won't be importable until we add src/ to sys.path ourselves (invalidate_caches
            # does not re-read .pth files). A fresh `python` picks it up automatically.
            _src = str(_repo_root / "src")
            if _src not in sys.path:
                sys.path.insert(0, _src)
            _log.append(f"✓ gen_gec_errant: added {_src} to sys.path (in-process editable import)")
        elif IN_COLAB:
            _url = "git+https://github.com/berstearns/gen-gec-errant-public.git"
            _log.append(f"… gen_gec_errant: pip install {_url}")
            _pip(_url)
        else:
            raise RuntimeError(
                "Could not find the repo root (pyproject.toml + src/gen_gec_errant). "
                "Run marimo from the repository root, or set IN_COLAB."
            )
        importlib.invalidate_caches()
        importlib.import_module("gen_gec_errant")
        _log.append("✓ gen_gec_errant: importable after install")

    # ERRANT is a declared dependency (pulled by -e .) but double-ensure it and its spaCy model.
    _ensure("errant", ["errant"])
    try:
        import spacy

        spacy.load("en_core_web_sm")
        _log.append("✓ en_core_web_sm: loadable")
    except Exception:
        _log.append("… en_core_web_sm: python -m spacy download en_core_web_sm")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        _log.append("✓ en_core_web_sm: downloaded")

    # Mount Google Drive on Colab (guarded so local runs skip it entirely).
    if IN_COLAB:
        try:
            from google.colab import drive

            drive.mount("/content/drive", force_remount=False)
            _log.append("✓ Google Drive mounted at /content/drive")
        except Exception as _e:  # noqa: F841
            _log.append(f"⚠ Drive mount skipped: {_e}")

    install_log = "\n".join(_log)
    return IN_COLAB, Path, install_log


@app.cell
def _(install_log):
    # Heavy + pipeline imports live in their own cell. Referencing `install_log` forces
    # marimo to order this AFTER Cell 0's install cell (marimo orders by data dependency),
    # so `import gen_gec_errant` below never runs before the pip install on a fresh env.
    _ = install_log
    import gc

    import matplotlib

    matplotlib.use("Agg")  # headless-safe; marimo renders the figure object regardless
    import matplotlib.pyplot as plt
    import torch

    # --- Reuse the REAL pipeline code (do not reimplement load / perplexity / GEC / ERRANT) ---
    from gen_gec_errant.annotation.config import AnnotationConfig
    from gen_gec_errant.annotation.runner import (
        ERRANTAnnotator,
        classify_errors_by_region,
    )
    from gen_gec_errant.colab import resolve_model_path
    from gen_gec_errant.data_loader.runner import make_prompts
    from gen_gec_errant.gec.config import GECConfig
    from gen_gec_errant.gec.runner import load_gec_corrector
    from gen_gec_errant.generation.config import GenerationParams, ModelConfig
    from gen_gec_errant.generation.runner import (
        compute_perplexity,
        get_device,
        load_model,
    )
    from gen_gec_errant.preprocessing.runner import split_into_sentences
    from gen_gec_errant.registry import MODEL_REGISTRY, PathConfig, PIPELINE_DEFAULTS

    DEVICE = get_device("auto")
    return (
        AnnotationConfig,
        DEVICE,
        ERRANTAnnotator,
        GECConfig,
        GenerationParams,
        MODEL_REGISTRY,
        ModelConfig,
        PIPELINE_DEFAULTS,
        PathConfig,
        classify_errors_by_region,
        compute_perplexity,
        gc,
        load_gec_corrector,
        load_model,
        make_prompts,
        plt,
        split_into_sentences,
        torch,
    )


@app.cell(hide_code=True)
def _(DEVICE, IN_COLAB, install_log, mo):
    # §8.2b reproducibility cell — the EXACT versions from the RUNNING interpreter, not >= ranges.
    import sys as _sys
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _ver

    def _v(name):
        try:
            return _ver(name)
        except PackageNotFoundError:
            return "— (not installed)"

    _pyver = _sys.version.split()[0]
    _lockname = f"requirements-lock-py{_sys.version_info.major}.{_sys.version_info.minor}.txt"
    _pkgs = {
        "python": _pyver,
        "torch": _v("torch"),
        "transformers": _v("transformers"),
        "errant": _v("errant"),
        "spacy": _v("spacy"),
        "en_core_web_sm": _v("en_core_web_sm"),
        "marimo": _v("marimo"),
        "numpy": _v("numpy"),
        "scipy": _v("scipy"),
        "pandas": _v("pandas"),
    }
    _rows = "\n".join(f"| `{k}` | `{val}` |" for k, val in _pkgs.items())
    mo.md(
        f"""
        **Environment resolved — reproducibility (§8.2b).**

        - `is_colab()` → **{IN_COLAB}** · resolved device → **`{DEVICE}`** *(CPU completes end-to-end,
          a few minutes/example; GPU used automatically when present).*
        - **Verified end-to-end on Python `{_pyver}`** with the *exact* versions below (not `>=`
          ranges). The frozen known-good set is **`notebooks/{_lockname}`** (a `pip freeze` of the
          verified CPU run).

        | package | exact version (running interpreter) |
        |---|---|
        {_rows}

        <details><summary>install log</summary>

        ```
        {install_log}
        ```
        </details>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Shared helpers (config tables, word diff, region colouring, acquisition view)
    *Canonical labels + the acquisition categories from CONCEPTS.md §1.2 live here.*
    """)
    return


@app.cell
def _(mo):
    # Canonical display labels for the three sources (CONCEPTS.md §1.1).
    SOURCE_LABELS = {
        "authentic_learner": "authentic learner",
        "matched_control": "matched control",
        "artificial_learner": "artificial learner (AL)",
    }
    SOURCE_ORDER = ["authentic_learner", "matched_control", "artificial_learner"]

    # SLA-diagnostic "acquisition categories" (CONCEPTS.md §1.2). "*" = any ERRANT operation (M/R/U).
    ACQUISITION_CATS = [
        ("*:DET", "determiner"),
        ("*:VERB:FORM", "verb morphology"),
        ("R:VERB:TENSE", "tense"),
        ("R:VERB:SVA", "subject–verb agreement"),
        ("*:PREP", "preposition"),
        ("R:NOUN:NUM", "noun number"),
    ]

    def tag_in_cat(tag: str, cat: str) -> bool:
        """Does an ERRANT tag fall in an acquisition category pattern (with '*' wildcard op)?"""
        if cat.startswith("*:"):
            suffix = cat[2:]
            parts = tag.split(":", 1)
            return len(parts) == 2 and parts[1] == suffix
        return tag == cat

    def acquisition_view(counts: dict) -> dict:
        """Map an error-tag count dict onto the six acquisition categories."""
        out = {}
        for cat, _name in ACQUISITION_CATS:
            out[cat] = sum(n for tag, n in counts.items() if tag_in_cat(tag, cat))
        return out

    def esc(s) -> str:
        import html as _h

        return _h.escape(str(s))

    def as_html(html_str: str):
        """Render a raw-HTML string in marimo, robust across versions."""
        try:
            return mo.Html(html_str)
        except Exception:
            return mo.md(html_str)

    def kv_table(title: str, d: dict, defaults: dict | None = None):
        """Render a dataclass-style config as a fields->values table, flagging non-default values."""
        rows = []
        for k, v in d.items():
            flag = ""
            if defaults is not None and k in defaults and defaults[k] != v:
                flag = f" <span style='color:#b26a00'>← differs from default `{esc(defaults[k])}`</span>"
            rows.append(f"| `{esc(k)}` | `{esc(v)}`{flag} |")
        body = "\n".join(rows)
        return mo.md(f"**{title}**\n\n| field | value |\n|---|---|\n{body}")

    def word_diff_html(original: str, corrected: str) -> str:
        """Word-level diff: deletions struck through (red), insertions green."""
        import difflib

        aw, bw = original.split(), corrected.split()
        sm = difflib.SequenceMatcher(a=aw, b=bw)
        parts = []
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == "equal":
                parts.append(esc(" ".join(aw[i1:i2])))
            elif op == "delete":
                parts.append(
                    f"<span style='background:#ffd6d6;text-decoration:line-through'>{esc(' '.join(aw[i1:i2]))}</span>"
                )
            elif op == "insert":
                parts.append(f"<span style='background:#d6f5d6'>{esc(' '.join(bw[j1:j2]))}</span>")
            elif op == "replace":
                parts.append(
                    f"<span style='background:#ffd6d6;text-decoration:line-through'>{esc(' '.join(aw[i1:i2]))}</span>"
                    f" <span style='background:#d6f5d6'>{esc(' '.join(bw[j1:j2]))}</span>"
                )
        return "<div style='line-height:1.9'>" + " ".join(parts) + "</div>"

    # --- Seed / left-context highlighting palette (Phase-1 viz) ---
    SEED_BG = "#fff3cd"  # seed / shared left context (the prompt)
    GEN_BG = "#d6f5d6"  # generated continuation (the source's own text)
    # Fixed canonical source palette, consistent across every figure (CONCEPTS.md order).
    SOURCE_PALETTE = {
        "authentic_learner": "#d1495b",  # yardstick — warm
        "matched_control": "#8d99ae",  # native-pretrained — muted
        "artificial_learner": "#2a9d8f",  # fine-tuned — teal
    }
    # SLA-relevant ERRANT tags the paper bolds in tab:error-type-distribution.
    SLA_TAGS = {"M:VERB:FORM", "R:VERB:SVA", "U:DET", "M:DET", "M:PRON"}

    def seed_legend():
        return mo.md(
            f"<span style='background:{SEED_BG};padding:1px 7px;border-radius:3px'>▧ seed "
            f"(shared left context)</span>&nbsp;&nbsp;"
            f"<span style='background:{GEN_BG};padding:1px 7px;border-radius:3px'>▧ generated</span>"
        )

    def seed_span(text):
        return f"<span style='background:{SEED_BG}'>{esc(text)}</span>"

    def gen_span(text):
        return f"<span style='background:{GEN_BG}'>{esc(text)}</span>"

    def seed_box(prompt_text):
        """The prominent shared-seed block, rendered once (all sources share it)."""
        return (
            f"<div><b>Seed / left context (shared by all sources)</b>"
            f"<div style='background:{SEED_BG};padding:7px 9px;border-radius:4px;line-height:1.7;margin-top:3px'>"
            f"{esc(prompt_text)}</div></div>"
        )

    def gen_box(text):
        """A generated continuation block (one source's own text)."""
        return (
            f"<div style='background:{GEN_BG};padding:7px 9px;border-radius:4px;line-height:1.7'>"
            f"{esc(text)}</div>"
        )

    def highlight_split(full_text, prompt_len):
        """Background the prompt slice [:prompt_len] as seed and the rest as generated
        (exact char split on the ORIGINAL full text — never on GEC-corrected text)."""
        return (
            "<div style='line-height:1.9'>"
            + seed_span(full_text[:prompt_len])
            + gen_span(full_text[prompt_len:])
            + "</div>"
        )

    return (
        ACQUISITION_CATS,
        SLA_TAGS,
        SOURCE_LABELS,
        SOURCE_PALETTE,
        acquisition_view,
        as_html,
        esc,
        gen_box,
        kv_table,
        seed_box,
        seed_legend,
        word_diff_html,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Cell 1 — Pick the example
    """)
    return


@app.cell
def _(mo):
    # Hardcoded fixture (Stage 0): full authentic-learner sentences copied VERBATIM from
    # data/celva-2-sample.csv (the `text` column). Each has >=1 visible learner error.
    # No file/dataset dependency is needed for the default run.
    EXAMPLES = [
        {
            "id": "celva-01",
            "text": "This magnet must be cooled by liquid nitrogen and that is the large cercle where we go through when we do an MRI.",
        },
        {
            "id": "celva-02",
            "text": "So with all of that we can make a matrix, with the hydrogen density and then convert it in a picture.",
        },
        {
            "id": "celva-03",
            "text": "The world have many riches at discover even too and its not necessary to have a Apple vision pro in addition.",
        },
        {
            "id": "celva-04",
            "text": "To do that we use electromagnetic flashs whitch are at the resonating frequency of hydrogen.",
        },
        {
            "id": "celva-05",
            "text": "Now with detectors we can calculate the among of time the spin needs to point to up again.",
        },
    ]

    example_dropdown = mo.ui.dropdown(
        options={f"{e['id']}: {e['text'][:60]}…": i for i, e in enumerate(EXAMPLES)},
        value=f"{EXAMPLES[0]['id']}: {EXAMPLES[0]['text'][:60]}…",
        label="Example (authentic-learner sentence)",
    )
    custom_text_area = mo.ui.text_area(
        value="",
        label="…or paste your OWN sentence (overrides the dropdown when non-empty)",
        full_width=True,
    )
    mo.vstack([example_dropdown, custom_text_area])
    return EXAMPLES, custom_text_area, example_dropdown


@app.cell
def _(EXAMPLES, custom_text_area, example_dropdown, mo, split_into_sentences):
    # Resolve the single working sentence. A pasted paragraph is reduced to its first
    # sentence (mirrors the batch data_loader's split_sentences=True behaviour).
    _custom = custom_text_area.value.strip()
    if _custom:
        chosen_source = "custom (pasted)"
        _raw = _custom
    else:
        _idx = example_dropdown.value if example_dropdown.value is not None else 0
        chosen_source = EXAMPLES[_idx]["id"]
        _raw = EXAMPLES[_idx]["text"]

    _sents = split_into_sentences(_raw)
    chosen_text = _sents[0] if _sents else _raw
    _note = ""
    if len(_sents) > 1:
        _note = f"\n\n> ℹ︎ input had {len(_sents)} sentences; using the **first** one (single-sentence pipeline)."

    mo.md(
        f"""
        **Chosen source:** `{chosen_source}`

        > {chosen_text}

        *(words: {len(chosen_text.split())}){_note}*
        """
    )
    return (chosen_text,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Cell 2 — STAGE 1 · data_loader: sentence → prompt + reference

    `make_prompts` splits the authentic sentence into a **prompt** (shared prefix the
    models continue) and a **reference** continuation. The reference **IS the authentic
    learner leg** — same sentence, real human continuation, error-profiled by the same
    instrument. The char boundary `len(prompt)` is what later separates *prompt-region*
    (shared, excluded) from *generation-region* (the source's own) errors.
    """)
    return


@app.cell
def _(PIPELINE_DEFAULTS, chosen_text, kv_table, make_prompts, mo):
    # Stage-1 config (DataLoaderConfig-style knobs). prompt_ratio=0.5, min_prompt_words=5.
    _cfg = {
        "prompt_ratio": 0.5,
        "min_prompt_words": 5,
        "split_sentences": True,
    }
    s1 = make_prompts([chosen_text], prompt_ratio=0.5, min_prompt_words=5)[0]
    prompt = s1["prompt"]
    reference = s1["reference"]
    boundary = len(prompt)

    _cfg_table = kv_table("DataLoaderConfig (this stage)", _cfg, PIPELINE_DEFAULTS["data_loader"])
    _split_table = mo.md(
        f"""
        | leg | text | words |
        |---|---|---|
        | **prompt** (shared, dim) | <span style='color:#888'>{prompt}</span> | {len(prompt.split())} |
        | **reference = authentic learner** | <span style='background:#fff3cd'>{reference}</span> | {len(reference.split())} |

        - **char boundary `len(prompt)` = {boundary}** — errors at char < {boundary} are *prompt-region* (excluded from every cross-source claim); char ≥ {boundary} are *generation-region* (counted).
        - The **reference continuation IS the authentic learner leg** — same sentence, real human continuation.
        """
    )
    mo.vstack([_cfg_table, _split_table])
    return boundary, prompt, reference


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Cell 3 — Model selection (the triad) — trivially switchable

    The demo pair is `ft-gpt2-small : gpt2` (AL : matched control) — identical architecture
    + size, differing **only** by fine-tuning on the learner corpus (EFCAMDAT). Swapping in
    another registry pair is a one-click change below, not a code edit.
    """)
    return


@app.cell
def _(MODEL_REGISTRY, mo):
    # One-click model switching. Defaults = the locked demo pair ft-gpt2-small : gpt2.
    al_reg_dropdown = mo.ui.dropdown(
        options=list(MODEL_REGISTRY.keys()),
        value="ft-gpt2-small",
        label="artificial learner (AL) — registry key",
    )
    control_hf_text = mo.ui.text(
        value="gpt2",
        label="matched control — HuggingFace id (loads everywhere)",
        full_width=True,
    )
    al_ckpt_text = mo.ui.text(value="", label="AL checkpoint dir override (blank = auto-resolve)", full_width=True)
    mo.vstack([al_reg_dropdown, control_hf_text, al_ckpt_text])
    return al_ckpt_text, al_reg_dropdown, control_hf_text


@app.cell
def _(
    IN_COLAB,
    MODEL_REGISTRY,
    ModelConfig,
    Path,
    PathConfig,
    al_ckpt_text,
    al_reg_dropdown,
    control_hf_text,
    mo,
    reference,
):
    # -- authentic learner: no model; its continuation = the Stage-1 reference.
    # -- matched control: a native HF id (gpt2), loads everywhere.
    control_cfg = ModelConfig(name="matched_control", hf_model_id=control_hf_text.value.strip() or "gpt2")

    # -- artificial learner: resolve the fine-tuned checkpoint dir via the real PathConfig
    #    (Colab Drive vs local), i.e. models_root / gdrive_subpath / checkpoint_subdir.
    _reg = MODEL_REGISTRY[al_reg_dropdown.value]
    _paths = PathConfig.for_colab() if IN_COLAB else PathConfig.for_local()
    _auto_ckpt = _paths.model_gdrive_path(_reg)  # None for native-only registry entries

    _override = al_ckpt_text.value.strip()
    al_ckpt_path = Path(_override) if _override else _auto_ckpt

    al_available = False
    al_load_kind = "none"
    al_cfg = None
    if al_ckpt_path is not None and al_ckpt_path.exists():
        if (al_ckpt_path / "config.json").exists():
            # HF-format checkpoint dir -> load by pointing hf_model_id at the directory.
            al_available = True
            al_load_kind = "hf_dir"
            al_cfg = ModelConfig(
                name="artificial_learner",
                hf_model_id=str(al_ckpt_path),
                model_family=_reg.model_family,
                is_learner_tuned=False,
            )
        else:
            _pt = [p for p in al_ckpt_path.glob("*.pt")] + [p for p in al_ckpt_path.glob("*.bin")]
            if _pt:
                al_available = True
                al_load_kind = "state_dict"
                al_cfg = ModelConfig(
                    name="artificial_learner",
                    hf_model_id="gpt2",
                    model_family=_reg.model_family,
                    is_learner_tuned=True,
                    checkpoint_path=str(_pt[0]),
                )

    if al_available:
        al_banner = mo.md(
            f"""
            ✅ **AL checkpoint found** — `{al_ckpt_path}` (load kind: `{al_load_kind}`).
            The AL leg will run. Registry entry: `{al_reg_dropdown.value}` — {_reg.description}.
            """
        )
    else:
        al_banner = mo.callout(
            mo.md(
                f"""
                **AL checkpoint not found at** `{al_ckpt_path}` — **AL leg skipped**;
                matched control + authentic learner still run. Point the override box above at a
                real checkpoint dir to enable it. (Expected artefacts: `config.json` + a weights
                file for an HF dir, or a `*.pt` / `*.bin` state-dict.)
                """
            ),
            kind="warn",
        )

    mo.vstack(
        [
            mo.md(
                f"""
                | source | role | model |
                |---|---|---|
                | **authentic learner** | yardstick | *(no model — continuation = Stage-1 reference: "{reference[:40]}…")* |
                | **matched control** | native-pretrained, same size | `{control_cfg.hf_model_id}` |
                | **artificial learner (AL)** | fine-tuned | `{al_reg_dropdown.value}` → `{al_ckpt_path}` |
                """
            ),
            al_banner,
        ]
    )
    return al_available, al_cfg, control_cfg


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Cell 2b — Generation parameters (§5: 1 → 50 tokens, stop at end of sentence)

    `min_new_tokens=1`, `max_new_tokens=50` (capped at 50), plus a **sentence-stop** so each
    continuation is exactly one sentence. **Notebook-only** — the batch pipeline defaults are
    untouched. Tweak and the generation re-runs reactively.
    """)
    return


@app.cell
def _(PIPELINE_DEFAULTS, mo):
    _g = PIPELINE_DEFAULTS["generation"]
    min_tokens_ui = mo.ui.number(start=1, stop=50, step=1, value=1, label="min_new_tokens")
    max_tokens_ui = mo.ui.number(start=1, stop=50, step=1, value=50, label="max_new_tokens (≤50)")
    temp_ui = mo.ui.number(start=0.1, stop=2.0, step=0.05, value=float(_g["temperature"]), label="temperature")
    topk_ui = mo.ui.number(start=0, stop=200, step=1, value=int(_g["top_k"]), label="top_k")
    topp_ui = mo.ui.number(start=0.1, stop=1.0, step=0.01, value=float(_g["top_p"]), label="top_p")
    rep_pen_ui = mo.ui.number(
        start=1.0, stop=2.0, step=0.05, value=float(_g["repetition_penalty"]), label="repetition_penalty"
    )
    do_sample_ui = mo.ui.checkbox(value=bool(_g["do_sample"]), label="do_sample")
    seed_ui = mo.ui.number(start=0, stop=999999, step=1, value=42, label="seed")
    mo.vstack(
        [
            mo.hstack([min_tokens_ui, max_tokens_ui, temp_ui, seed_ui], justify="start"),
            mo.hstack([topk_ui, topp_ui, rep_pen_ui, do_sample_ui], justify="start"),
        ]
    )
    return (
        do_sample_ui,
        max_tokens_ui,
        min_tokens_ui,
        rep_pen_ui,
        seed_ui,
        temp_ui,
        topk_ui,
        topp_ui,
    )


@app.cell
def _(torch):
    # §5 sentence-stop. This faithfully MIRRORS gen_gec_errant.generation.runner.generate_continuations
    # (same tokenization + same model.generate args) and ADDS a HuggingFace StoppingCriteria that
    # halts once the newly-generated continuation contains a sentence terminator. Kept in the
    # notebook (not in src/) per SPEC §5 so no batch-pipeline source is touched. Designed for a
    # single prompt (batch_size=1), which is how the notebook calls it.
    def generate_with_sentence_stop(model, tokenizer, prompt_text, gen_params, device, terminators=(".", "!", "?")):
        from transformers import StoppingCriteria, StoppingCriteriaList

        inputs = tokenizer(
            [prompt_text], return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)
        prompt_len = int(inputs["attention_mask"].sum().item())

        class _SentenceStop(StoppingCriteria):
            def __call__(self, input_ids, scores=None, **kwargs):
                done = []
                for row in range(input_ids.shape[0]):
                    text = tokenizer.decode(input_ids[row, prompt_len:], skip_special_tokens=True)
                    done.append(any(t in text for t in terminators))
                return torch.tensor(done, dtype=torch.bool, device=input_ids.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=gen_params.max_new_tokens,
                min_new_tokens=gen_params.min_new_tokens,
                temperature=gen_params.temperature,
                top_k=gen_params.top_k,
                top_p=gen_params.top_p,
                do_sample=gen_params.do_sample,
                num_return_sequences=gen_params.num_return_sequences,
                repetition_penalty=gen_params.repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                stopping_criteria=StoppingCriteriaList([_SentenceStop()]),
            )
        gen_ids = out[0, prompt_len:]
        raw = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return raw, int(gen_ids.shape[0])

    return (generate_with_sentence_stop,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Cell 4 — STAGE 2 · generation (per model source)

    For **matched control** and **AL**: `load_model` → generate (1→50 tokens, sentence-stop)
    → trim to the first sentence via `split_into_sentences(...)[0]` → perplexity on the full
    text. The **authentic learner** needs no generation — its continuation is the reference.
    Each model is freed (`del; gc; empty_cache`) before the next loads — mirroring the batch
    runner's memory discipline. Both the **raw** (pre-trim) and **final one-sentence** outputs
    are shown so the sentence-stop's effect is visible.
    """)
    return


@app.cell
def _(
    DEVICE,
    GenerationParams,
    PIPELINE_DEFAULTS,
    SOURCE_LABELS,
    al_available,
    al_cfg,
    as_html,
    compute_perplexity,
    control_cfg,
    do_sample_ui,
    gc,
    gen_box,
    generate_with_sentence_stop,
    kv_table,
    load_model,
    max_tokens_ui,
    min_tokens_ui,
    mo,
    prompt,
    reference,
    rep_pen_ui,
    seed_box,
    seed_legend,
    seed_ui,
    split_into_sentences,
    temp_ui,
    topk_ui,
    topp_ui,
    torch,
):
    gen_params_used = GenerationParams(
        max_new_tokens=int(max_tokens_ui.value),
        min_new_tokens=int(min_tokens_ui.value),
        temperature=float(temp_ui.value),
        top_k=int(topk_ui.value),
        top_p=float(topp_ui.value),
        do_sample=bool(do_sample_ui.value),
        repetition_penalty=float(rep_pen_ui.value),
    )

    def _run_generation():
        results = {}
        # authentic learner: no model, continuation = reference.
        results["authentic_learner"] = {
            "continuation_raw": reference,
            "continuation": reference,
            "full_text": f"{prompt} {reference}",
            "n_new_tokens": None,
            "perplexity": None,
            "model_class": "— (no model; authentic human continuation)",
            "param_count": None,
        }

        _to_run = [("matched_control", control_cfg)]
        if al_available and al_cfg is not None:
            _to_run.append(("artificial_learner", al_cfg))

        for _name, _cfg in _to_run:
            torch.manual_seed(int(seed_ui.value))
            model, tok = load_model(_cfg, DEVICE)
            _cls = type(model).__name__
            _params = sum(p.numel() for p in model.parameters())
            # Weight fingerprint (mean |token-embedding|). The AL (fine-tuned) and the matched
            # control (base gpt2) share an architecture, so an IDENTICAL fingerprint would mean
            # the AL silently fell back to base gpt2. A DIFFERENT fingerprint proves it is the
            # fine-tune actually loaded from the checkpoint (§8.4).
            try:
                _wte_fp = float(model.get_input_embeddings().weight.detach().float().abs().mean().item())
            except Exception:
                _wte_fp = None
            raw, n_new = generate_with_sentence_stop(model, tok, prompt, gen_params_used, DEVICE)
            _sents = split_into_sentences(raw)
            trimmed = _sents[0] if _sents else raw
            full_text = f"{prompt} {trimmed}"
            ppl = compute_perplexity(model, tok, [full_text], batch_size=1, device=DEVICE)[0]
            results[_name] = {
                "continuation_raw": raw,
                "continuation": trimmed,
                "full_text": full_text,
                "n_new_tokens": n_new,
                "perplexity": ppl,
                "model_class": _cls,
                "param_count": _params,
                "source_path": _cfg.hf_model_id,
                "wte_fp": _wte_fp,
            }
            # Free before loading the next model (mirror the batch runner).
            del model, tok
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
        return results

    sources = _run_generation()

    def _render():
        # Seed / left context is IDENTICAL for all three sources — render it ONCE, prominently.
        blocks = [
            seed_legend(),
            as_html(seed_box(prompt)),
            mo.md(
                "*Generation stops at the model's first sentence terminator (one sentence); the "
                "GEC step (Stage 3) may then split a run-on into several corrected sentences.*"
            ),
            kv_table(
                "GenerationParams (this stage — notebook-only sentence-stop)",
                {
                    "min_new_tokens": gen_params_used.min_new_tokens,
                    "max_new_tokens": gen_params_used.max_new_tokens,
                    "temperature": gen_params_used.temperature,
                    "top_k": gen_params_used.top_k,
                    "top_p": gen_params_used.top_p,
                    "do_sample": gen_params_used.do_sample,
                    "repetition_penalty": gen_params_used.repetition_penalty,
                    "seed": int(seed_ui.value),
                },
                {**PIPELINE_DEFAULTS["generation"], "seed": 42},
            ),
        ]
        # Per source, show only what DIFFERS: its own generated continuation (green).
        for _name in ["authentic_learner", "matched_control", "artificial_learner"]:
            if _name not in sources:
                continue
            d = sources[_name]
            _ppl = f"{d['perplexity']:.2f}" if d["perplexity"] is not None else "— (human, no PPL)"
            _pc = f"{d['param_count']:,}" if d["param_count"] is not None else "—"
            _nt = d["n_new_tokens"] if d["n_new_tokens"] is not None else "—"
            _fp = d.get("wte_fp")
            _fp_str = f"`{_fp:.6f}`" if _fp is not None else "—"
            _src = d.get("source_path", "—")
            if _name == "authentic_learner":
                _info = f"- **human continuation** — this IS the authentic learner leg (no model); perplexity {_ppl}."
            else:
                _info = (
                    f"- **loaded model:** `{d['model_class']}` · params {_pc} · loaded from `{_src}`\n"
                    f"- new tokens: **{_nt}** (≤ {gen_params_used.max_new_tokens}, sentence-stop) · "
                    f"perplexity {_ppl} · token-embedding fingerprint {_fp_str}"
                )
            _sub = []
            if _name != "authentic_learner" and d["continuation_raw"] != d["continuation"]:
                _sub.append(
                    mo.md(
                        f"raw pre-trim (before the one-sentence cut): "
                        f"<span style='color:#777'>{d['continuation_raw']}</span>"
                    )
                )
            blocks.append(
                mo.vstack(
                    [
                        mo.md(f"### {SOURCE_LABELS[_name]}"),
                        mo.md(_info),
                        mo.md("**generated continuation** (this source's own text — follows the shared seed above):"),
                        as_html(gen_box(d["continuation"])),
                        *_sub,
                    ]
                )
            )
        return mo.vstack(blocks)

    _render()
    return (sources,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Cell 5 — STAGE 3 · GEC (per source)

    The GEC model (`grammarly/coedit-large`, `method="dedicated"`, beam search) is loaded
    **once** and reused. For each source we correct `full_text = prompt + " " + continuation`.
    The corrected text is **what the pipeline treats as the error-free target** — the
    difference between original and corrected is exactly where the errors are.
    """)
    return


@app.cell
def _(DEVICE, GECConfig, load_gec_corrector, mo, sources):
    # Load the corrector ONCE, reuse for every source.
    gec_config = GECConfig()  # defaults: dedicated, grammarly/coedit-large, num_beams=4
    _corrector = load_gec_corrector(gec_config, DEVICE)

    gec = {}
    for _name, _d in sources.items():
        _corrected = _corrector.correct([_d["full_text"]])[0]
        gec[_name] = {"full_text": _d["full_text"], "corrected": _corrected}

    del _corrector
    _msg = mo.md(f"GEC corrector `{gec_config.model_id}` loaded once; corrected {len(gec)} sources.")
    _msg
    return gec, gec_config


@app.cell
def _(SOURCE_LABELS, as_html, gec, gec_config, kv_table, mo, word_diff_html):
    def _render():
        cfg = kv_table(
            "GECConfig (this stage)",
            {"method": gec_config.method, "model_id": gec_config.model_id, "num_beams": 4, "device": "auto"},
        )
        blocks = [
            cfg,
            mo.md(
                "GEC corrects the **full text** (shared seed + this source's continuation). The word-diff "
                "marks <span style='background:#ffd6d6'>deletions</span> / "
                "<span style='background:#d6f5d6'>insertions</span>. Only **generation-region** edits "
                "(Stage 4) are counted; prompt-region edits are shared and excluded. *GEC may split a "
                "run-on continuation into several corrected sentences — that is why the corrected text can "
                "have more sentence breaks than the one-sentence generation.*"
            ),
        ]
        for _name in ["authentic_learner", "matched_control", "artificial_learner"]:
            if _name not in gec:
                continue
            d = gec[_name]
            blocks.append(
                mo.vstack(
                    [
                        mo.md(f"### {SOURCE_LABELS[_name]}"),
                        mo.md("**original → corrected (word-level diff):**"),
                        as_html(word_diff_html(d["full_text"], d["corrected"])),
                        mo.md(f"**corrected (error-free target):** {d['corrected']}"),
                    ]
                )
            )
        return mo.vstack(blocks)

    _render()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Cell 6 — STAGE 4 · ERRANT (per source, region-classified)

    ERRANT tags every correction edit. We annotate the **full text** then call
    `classify_errors_by_region([ann], [len(prompt)])`: edits at char < boundary are
    **prompt-region** (shared, *excluded*, greyed) and char ≥ boundary are
    **generation-region** (the source's own, *counted*, solid). The per-source
    `generation_error_type_counts` is the error-tag distribution every comparison uses.
    """)
    return


@app.cell
def _(ERRANTAnnotator):
    # Load the ERRANT annotator ONCE and reuse across sources.
    errant_annotator = ERRANTAnnotator("en")
    return (errant_annotator,)


@app.cell
def _(
    AnnotationConfig,
    boundary,
    classify_errors_by_region,
    errant_annotator,
    gec,
    kv_table,
):
    annos = {}
    gen_tag_counts = {}
    for _name, _d in gec.items():
        _ann = errant_annotator.annotate_pair(_d["full_text"], _d["corrected"])
        classify_errors_by_region([_ann], [boundary])
        annos[_name] = _ann
        gen_tag_counts[_name] = dict(_ann.generation_error_type_counts)

    _acfg = AnnotationConfig(lang="en")
    ann_cfg_table = kv_table("AnnotationConfig (this stage)", {"lang": _acfg.lang, "prompt_char_boundary": boundary})
    return ann_cfg_table, annos, gen_tag_counts


@app.cell
def _(
    SOURCE_LABELS,
    ann_cfg_table,
    annos,
    as_html,
    boundary,
    esc,
    gen_tag_counts,
    mo,
):
    def _edit_table(ann):
        rows = [
            "<tr><th>orig tokens</th><th>→ corr tokens</th><th>ERRANT tag</th>"
            "<th>char span</th><th>region</th></tr>"
        ]
        if not ann.errors:
            rows.append("<tr><td colspan=5><i>no edits (GEC left it unchanged)</i></td></tr>")
        for e in ann.errors:
            counted = e.region == "generation"
            bg = "#ffffff" if counted else "#efefef"
            col = "#111" if counted else "#999"
            tag_style = "font-weight:700" if counted else "font-weight:400"
            rows.append(
                f"<tr style='background:{bg};color:{col}'>"
                f"<td>{esc(e.original_tokens) or '∅'}</td>"
                f"<td>{esc(e.corrected_tokens) or '∅'}</td>"
                f"<td style='{tag_style}'>{esc(e.error_type)}</td>"
                f"<td>{e.char_start}–{e.char_end}</td>"
                f"<td>{esc(e.region)}{' ✓counted' if counted else ' (excluded)'}</td></tr>"
            )
        return (
            "<table style='border-collapse:collapse' border=1 cellpadding=4>"
            + "".join(rows)
            + "</table>"
        )

    def _render():
        blocks = [
            ann_cfg_table,
            mo.md(
                f"*Grey rows = prompt-region (char < {boundary}, excluded). "
                f"Solid rows = generation-region (char ≥ {boundary}, counted).*"
            ),
        ]
        for _name in ["authentic_learner", "matched_control", "artificial_learner"]:
            if _name not in annos:
                continue
            ann = annos[_name]
            _dist = gen_tag_counts[_name]
            _dist_str = ", ".join(f"`{t}`×{n}" for t, n in sorted(_dist.items())) or "*(none)*"
            blocks.append(
                mo.vstack(
                    [
                        mo.md(f"### {SOURCE_LABELS[_name]}"),
                        as_html(_edit_table(ann)),
                        mo.md(
                            f"**generation-region error-tag distribution** "
                            f"(`ann.generation_error_type_counts`): {_dist_str}"
                        ),
                    ]
                )
            )
        return mo.vstack(blocks)

    _render()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Cell 7 — STAGE 5 · paper-mirrored analysis (vs the authentic learner)

    Mirrors the paper's tables/figures with the **canonical three sources** (authentic learner ·
    matched control · artificial learner (AL)), all on the **generation-region** tags. **n=1**:
    every value the paper aggregates over 20 sentences is a *single* number here — labelled *this
    sentence*, never avg/rate. Framing stays *distance / closeness to the authentic learner*.
    """)
    return


@app.cell
def _(SOURCE_LABELS, as_html, esc, gen_tag_counts, mo, sources):
    # 1b.1 — Error-metrics table (mirrors paper tab:error-metrics), 3 sources + Δ(AL − control).
    def _metrics_table():
        cols = [n for n in ["authentic_learner", "matched_control", "artificial_learner"] if n in gen_tag_counts]
        _err = {c: sum(gen_tag_counts[c].values()) for c in cols}
        _ppl = {c: sources[c]["perplexity"] for c in cols}
        _has_delta = "matched_control" in cols and "artificial_learner" in cols

        head = (
            "<tr><th>metric (this sentence, n=1)</th>"
            + "".join(f"<th>{esc(SOURCE_LABELS[c])}</th>" for c in cols)
            + ("<th>Δ (AL − control)</th>" if _has_delta else "")
            + "</tr>"
        )
        rows = [head]

        # full-text perplexity (authentic learner has none)
        _pcell = {c: (f"{_ppl[c]:.2f}" if _ppl[c] is not None else "— (human)") for c in cols}
        _dppl = ""
        if _has_delta:
            _a, _c = _ppl["artificial_learner"], _ppl["matched_control"]
            _dppl = (
                f"<td>{_a - _c:+.2f} ({(_a - _c) / _c * 100:+.1f}%)</td>"
                if (_a is not None and _c) else "<td>—</td>"
            )
        rows.append(
            "<tr><td><b>full-text perplexity</b></td>"
            + "".join(f"<td>{_pcell[c]}</td>" for c in cols) + _dppl + "</tr>"
        )

        # total generation-region errors (this sentence)
        _derr = f"<td>{_err['artificial_learner'] - _err['matched_control']:+d}</td>" if _has_delta else ""
        rows.append(
            "<tr><td><b>total generation-region errors</b> <span style='color:#666'>(this sentence)</span></td>"
            + "".join(f"<td>{_err[c]}</td>" for c in cols) + _derr + "</tr>"
        )

        # has >=1 generation-region error
        rows.append(
            "<tr><td><b>has ≥1 generation-region error</b></td>"
            + "".join(f"<td>{'yes' if _err[c] else 'no'}</td>" for c in cols)
            + ("<td>—</td>" if _has_delta else "") + "</tr>"
        )
        return "<table style='border-collapse:collapse' border=1 cellpadding=5>" + "".join(rows) + "</table>"

    mo.vstack([
        mo.md(
            "**Error metrics** — mirrors paper `tab:error-metrics` (paper: mean PPL±σ, total & avg "
            "errors, error rate over 20 sentences → here a single sentence, so *total errors this "
            "sentence* and *has ≥1*, not avg/rate)."
        ),
        as_html(_metrics_table()),
    ])
    return


@app.cell
def _(
    ACQUISITION_CATS,
    SLA_TAGS,
    SOURCE_LABELS,
    acquisition_view,
    as_html,
    esc,
    gen_tag_counts,
    mo,
):
    # 1b.2 — Error-type distribution table (mirrors paper tab:error-type-distribution) + acquisition view.
    def _cols():
        return [n for n in ["authentic_learner", "matched_control", "artificial_learner"] if n in gen_tag_counts]

    def _dist_table():
        cols = _cols()
        all_tags = sorted(
            {t for c in cols for t in gen_tag_counts[c]},
            key=lambda t: (-sum(gen_tag_counts[c].get(t, 0) for c in cols), t),
        )
        head = "<tr><th>ERRANT tag</th>" + "".join(f"<th>{esc(SOURCE_LABELS[c])}</th>" for c in cols) + "</tr>"
        rows = [head]
        if not all_tags:
            rows.append(f"<tr><td colspan={len(cols) + 1}><i>no generation-region errors on any source (this sentence)</i></td></tr>")
        for t in all_tags:
            _sla = t in SLA_TAGS
            _bg = "background:#fff8e1" if _sla else ""
            _tag = f"<b>{esc(t)}</b>" if _sla else esc(t)
            rows.append(
                f"<tr style='{_bg}'><td>{_tag}</td>"
                + "".join(
                    (f"<td><b>{gen_tag_counts[c].get(t, 0)}</b></td>" if _sla else f"<td>{gen_tag_counts[c].get(t, 0)}</td>")
                    for c in cols
                )
                + "</tr>"
            )
        return "<table style='border-collapse:collapse' border=1 cellpadding=5>" + "".join(rows) + "</table>"

    def _acq_table():
        cols = _cols()
        head = "<tr><th>acquisition category</th>" + "".join(f"<th>{esc(SOURCE_LABELS[c])}</th>" for c in cols) + "</tr>"
        rows = [head]
        for cat, name in ACQUISITION_CATS:
            rows.append(
                f"<tr><td><code>{esc(cat)}</code> <span style='color:#666'>({esc(name)})</span></td>"
                + "".join(f"<td>{acquisition_view(gen_tag_counts[c]).get(cat, 0)}</td>" for c in cols)
                + "</tr>"
            )
        return "<table style='border-collapse:collapse' border=1 cellpadding=5>" + "".join(rows) + "</table>"

    mo.vstack([
        mo.md(
            "**Error-type distribution** — mirrors paper `tab:error-type-distribution`. Rows ordered "
            "by total count; **bold + shaded rows** are the SLA-relevant tags the paper highlights "
            "(`M:VERB:FORM`, `R:VERB:SVA`, `U:DET`, `M:DET`, `M:PRON`)."
        ),
        as_html(_dist_table()),
        mo.md("**Acquisition-category view** (CONCEPTS.md §1.2 — the six SLA-diagnostic buckets):"),
        as_html(_acq_table()),
    ])
    return


@app.cell
def _(SOURCE_LABELS, SOURCE_PALETTE, gen_tag_counts, mo, plt):
    # 1b.3 — Error-type bar chart (mirrors paper fig:error-type-breakdown), ordered by total desc.
    def _chart():
        cols = [n for n in ["authentic_learner", "matched_control", "artificial_learner"] if n in gen_tag_counts]
        import numpy as np

        all_tags = sorted(
            {t for c in cols for t in gen_tag_counts[c]},
            key=lambda t: -sum(gen_tag_counts[c].get(t, 0) for c in cols),
        )
        if not all_tags:
            return mo.md("*No generation-region error tags on any source — nothing to plot (mirrors `fig:error-type-breakdown`).*")
        x = np.arange(len(all_tags))
        width = 0.8 / max(len(cols), 1)
        fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(all_tags)), 4))
        for i, c in enumerate(cols):
            ax.bar(x + i * width, [gen_tag_counts[c].get(t, 0) for t in all_tags], width,
                   label=SOURCE_LABELS[c], color=SOURCE_PALETTE[c])
        ax.set_xticks(x + width * (len(cols) - 1) / 2)
        ax.set_xticklabels(all_tags, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("generation-region count (this sentence)")
        ax.set_title("Error-type breakdown — 3 sources (n=1) · mirrors fig:error-type-breakdown")
        ax.legend(fontsize=8)
        ax.margins(y=0.15)
        fig.tight_layout()
        return fig

    _chart()
    return


@app.cell
def _(SOURCE_LABELS, SOURCE_PALETTE, mo, plt, sources):
    # 1b.4 — Perplexity comparison bar (mirrors paper fig:perplexity-comparison). Authentic learner: n/a.
    def _chart():
        cols = [n for n in ["authentic_learner", "matched_control", "artificial_learner"] if n in sources]
        vals = [(c, sources[c]["perplexity"]) for c in cols if sources[c]["perplexity"] is not None]
        if not vals:
            return mo.md("*No model perplexities to plot.*")
        fig, ax = plt.subplots(figsize=(5, 3.6))
        ax.bar(range(len(vals)), [v for _, v in vals], color=[SOURCE_PALETTE[c] for c, _ in vals])
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels([SOURCE_LABELS[c] for c, _ in vals], rotation=12, ha="right", fontsize=8)
        ax.set_ylabel("full-text perplexity")
        ax.set_title("Perplexity comparison (n=1) · mirrors fig:perplexity-comparison\n(authentic learner: n/a — human text, no model)", fontsize=9)
        for _i, (_, _v) in enumerate(vals):
            ax.text(_i, _v, f"{_v:.1f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        return fig

    _chart()
    return


@app.cell
def _(SOURCE_LABELS, gen_tag_counts, mo, sources):
    # 1b.5 — reading (distance to the authentic learner) + combined metric + scatter-omission note.
    def _reading():
        cols = [n for n in ["authentic_learner", "matched_control", "artificial_learner"] if n in gen_tag_counts]
        learner = set(gen_tag_counts.get("authentic_learner", {}))
        control = set(gen_tag_counts.get("matched_control", {}))
        if "artificial_learner" not in cols:
            return mo.callout(
                mo.md(
                    "**AL leg not available on this run** — closeness *to the authentic learner* can't "
                    "be measured. Compare **matched control** vs **authentic learner**: shared "
                    "generation-region tags are where even the matched control overlaps the yardstick."
                ),
                kind="info",
            )
        al = set(gen_tag_counts.get("artificial_learner", {}))
        _t = ", ".join(f"`{t}`" for t in sorted((learner & al) - control)) or "*(none on this single sentence)*"
        _s = ", ".join(f"`{t}`" for t in sorted(learner & al & control)) or "*(none)*"
        _cm = []
        for c in cols:
            _p = sources[c]["perplexity"]
            _e = sum(gen_tag_counts[c].values())
            _cm.append(f"- {SOURCE_LABELS[c]}: " + (f"{_p:.1f} × {_e} = **{_p * _e:.1f}**" if _p is not None else "— (human, no PPL)"))
        _cm_str = "\n".join(_cm)
        return mo.md(
            f"""
            **Reading — closeness to the authentic learner (the yardstick):**

            - Generation-region tags the **AL shares with the authentic learner that the control lacks**
              (the "moves *toward* the authentic learner" signal): {_t}
            - Tags shared by all three (overlap even native pretraining reaches): {_s}

            **Combined metric (PPL × errors)** — mirrors paper `fig:combined-metrics`; illustrative at n=1:

            {_cm_str}

            *Per-sentence PPL-vs-errors scatter (`fig:ppl-vs-errors`) is **omitted** — at n=1 it is a single
            point, so it is meaningless here.*

            On one sentence this is a handful of tags — *illustrative of the mechanism*, not a measurement.
            The claim `JSD(AL, LEARNER) < JSD(CONTROL, LEARNER)` is distributional over many sentences.
            """
        )

    _reading()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Cell 8 — Honest n=1 caveat (required)
    """)
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            """
            **This notebook is a single-example DEBUGGER for ownership — not evidence.**

            On `n=1`, a "distribution" is just a handful of tags: **illustrative only**. The
            central claim — `JSD(AL, LEARNER) < JSD(CONTROL, LEARNER)`, concentrated in
            acquisition categories — and its effect sizes (**JSD**, **RDR**) are **distributional
            over many sentences**. A single sentence can neither support nor refute it.

            For the real, pre-registered measurement run the **batch pipeline**:

            ```
            python -m gen_gec_errant.pipeline --config <config.yaml>
            ```

            and see `gen-gec-review-specs/` (claim, pre-registration, matrix, analyses) and
            `analysis-outputs/`. Do not read one sentence as if it settled anything.
            """
        ),
        kind="warn",
    )
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
