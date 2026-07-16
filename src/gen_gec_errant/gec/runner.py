"""GEC runner: correct generated text using LLM or dedicated GEC model."""

import logging
import re
import time
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from gen_gec_errant.gec.config import GECConfig

logger = logging.getLogger(__name__)


def _get_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def _format_eta(seconds: float) -> str:
    if seconds == float("inf") or seconds != seconds:  # inf or NaN
        return "unknown"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def correct_in_batches(corrector, sentences: List[str], batch_size: int, label: str) -> List[str]:
    """Run corrector.correct in batches, logging progress every batch.

    A single GEC batch (beam search over a seq2seq model) can take minutes on
    CPU; with no per-batch signal a multi-hour stage looks identical whether
    it's 10% or 90% done. Log every batch — batches are the coarse unit here,
    not thousands of fast steps, so this isn't spammy.
    """
    total = len(sentences)
    if total == 0:
        return []

    n_batches = (total + batch_size - 1) // batch_size
    corrected: List[str] = []
    start = time.monotonic()

    for batch_idx, i in enumerate(range(0, total, batch_size), start=1):
        batch = sentences[i : i + batch_size]
        corrected.extend(corrector.correct(batch))

        elapsed = time.monotonic() - start
        done = len(corrected)
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else float("inf")
        logger.info(
            "  [%s] batch %d/%d — %d/%d sentences (%.3f sents/s, ETA %s)",
            label, batch_idx, n_batches, done, total, rate, _format_eta(eta),
        )

    return corrected


class LLMCorrector:
    """GEC using an instruction-tuned LLM (e.g., Gemma)."""

    def __init__(self, config: GECConfig, device: torch.device):
        self.config = config
        self.device = device

        logger.info("Loading LLM corrector: %s", config.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        ).to(device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def correct(self, sentences: List[str]) -> List[str]:
        corrected = []
        for sent in sentences:
            prompt = self.config.prompt_template.format(sentence=sent)

            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(self.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=len(sent.split()) + 20,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            prompt_len = inputs["input_ids"].shape[1]
            result = self.tokenizer.decode(
                outputs[0][prompt_len:], skip_special_tokens=True
            ).strip()

            result = result.split("\n")[0].strip()
            result = re.split(r"(?:Explanation|Note|Reason):", result)[0].strip()

            if not result or len(result) < 3:
                result = sent

            corrected.append(result)
        return corrected


class DedicatedGECCorrector:
    """GEC using a purpose-built seq2seq model (e.g., coedit-large)."""

    def __init__(self, config: GECConfig, device: torch.device):
        self.config = config
        self.device = device

        logger.info("Loading dedicated GEC model: %s", config.model_id)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            config.model_id, torch_dtype=dtype,
        ).to(device)
        self.model.eval()

    @torch.no_grad()
    def correct(self, sentences: List[str]) -> List[str]:
        if not sentences:
            return []

        input_texts = [f"Fix grammatical errors in this sentence: {s}" for s in sentences]

        inputs = self.tokenizer(
            input_texts, return_tensors="pt", truncation=True,
            max_length=512, padding=True,
        ).to(self.device)

        max_tok = max(len(s.split()) for s in sentences) + 20

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tok,
            num_beams=4,
        )

        corrected = []
        for i, sent in enumerate(sentences):
            result = self.tokenizer.decode(outputs[i], skip_special_tokens=True).strip()
            if not result or len(result) < 3:
                result = sent
            corrected.append(result)
        return corrected


def load_gec_corrector(config: GECConfig, device: torch.device):
    """Factory to load the appropriate GEC corrector."""
    if config.method == "llm":
        return LLMCorrector(config, device)
    elif config.method == "dedicated":
        return DedicatedGECCorrector(config, device)
    else:
        raise ValueError(f"Unknown GEC method: {config.method}")


def run_gec(config: GECConfig, generation_results: dict) -> dict:
    """
    Run GEC on generation results.

    Args:
        config: GECConfig
        generation_results: Dict with keys continuations, full_texts, etc.

    Returns:
        Updated dict with corrected_continuations and corrected_full_texts added.
    """
    device = _get_device(config.device)
    corrector = load_gec_corrector(config, device)

    logger.info("Correcting outputs (method=%s)...", config.method)

    corrected_continuations = correct_in_batches(
        corrector, generation_results["continuations"], config.batch_size, "continuations")
    corrected_full_texts = correct_in_batches(
        corrector, generation_results["full_texts"], config.batch_size, "full_texts")

    generation_results["corrected_continuations"] = corrected_continuations
    generation_results["corrected_full_texts"] = corrected_full_texts

    logger.info("Corrected %d sentences", len(corrected_continuations))

    del corrector
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return generation_results
