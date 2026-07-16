"""Data loader runner: load sentences and prepare prompt/reference pairs."""

import csv
import logging
from pathlib import Path
from typing import List, Optional

from gen_gec_errant.data_loader.config import DataLoaderConfig

logger = logging.getLogger(__name__)


def load_sentences(
    path: str,
    max_sentences: Optional[int] = None,
    min_words: int = 10,
    max_words: int = 10000,
    text_column: Optional[str] = None,
) -> List[str]:
    """Load and filter sentences from a .txt, .csv, or .tsv file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    raw_sentences: list[str] = []

    if path.suffix == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            raw_sentences = [line.strip() for line in f if line.strip()]

    elif path.suffix in (".csv", ".tsv"):
        delimiter = "\t" if path.suffix == ".tsv" else ","
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            text_col = text_column
            for row in reader:
                if text_col is None:
                    for candidate in ["sentence", "text", "original_text",
                                      "writing", "learner_text", "corrected"]:
                        if candidate in row and row[candidate].strip():
                            text_col = candidate
                            logger.info(f"Auto-detected text column: '{text_col}'")
                            break
                    if text_col is None:
                        text_col = max(row.keys(), key=lambda k: len(str(row.get(k, ""))))
                        logger.info(f"Fallback text column: '{text_col}'")

                text = row.get(text_col, "").strip()
                if text:
                    raw_sentences.append(text)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    sentences = [
        s for s in raw_sentences
        if min_words <= len(s.split()) <= max_words
    ]

    logger.info(f"Loaded {len(raw_sentences)} raw sentences, {len(sentences)} after filtering")

    if max_sentences is not None:
        sentences = sentences[:max_sentences]
        logger.info(f"Capped to {max_sentences} sentences")

    return sentences


def make_prompts(
    sentences: List[str],
    prompt_ratio: float = 0.5,
    min_prompt_words: int = 3,
) -> List[dict]:
    """Split each sentence into a prompt prefix and a reference continuation."""
    items = []
    for sent in sentences:
        words = sent.split()
        split_idx = max(min_prompt_words, int(len(words) * prompt_ratio))
        if split_idx >= len(words):
            split_idx = len(words) - 1

        prompt = " ".join(words[:split_idx])
        reference = " ".join(words[split_idx:])

        items.append({
            "prompt": prompt,
            "reference": reference,
            "full": sent,
        })

    return items


def run_data_loader(config: DataLoaderConfig) -> List[dict]:
    """
    Run the data loading stage.

    Returns:
        List of dicts with keys: prompt, reference, full, text_id, sentence_idx
    """
    logger.info("Loading data from %s", config.data_path)

    if config.split_sentences:
        # Load all raw texts without filtering (filter at sentence level)
        raw_texts = load_sentences(
            path=config.data_path,
            max_sentences=None,
            min_words=0,
            max_words=999999,
            text_column=config.text_column,
        )

        from gen_gec_errant.preprocessing.runner import split_into_sentences

        sentences_meta = []
        for text_id, text in enumerate(raw_texts):
            for sent_idx, sent in enumerate(split_into_sentences(text)):
                wc = len(sent.split())
                if config.min_words <= wc <= config.max_words:
                    sentences_meta.append({
                        "text": sent,
                        "text_id": text_id,
                        "sentence_idx": sent_idx,
                    })

        logger.info(
            "Split %d texts into %d sentences (after word-count filter)",
            len(raw_texts), len(sentences_meta),
        )

        if config.max_sentences is not None:
            sentences_meta = sentences_meta[:config.max_sentences]
            logger.info("Capped to %d sentences", config.max_sentences)

        items = []
        for meta in sentences_meta:
            words = meta["text"].split()
            split_idx = max(config.min_prompt_words, int(len(words) * config.prompt_ratio))
            if split_idx >= len(words):
                split_idx = len(words) - 1
            items.append({
                "prompt": " ".join(words[:split_idx]),
                "reference": " ".join(words[split_idx:]),
                "full": meta["text"],
                "text_id": meta["text_id"],
                "sentence_idx": meta["sentence_idx"],
            })
    else:
        sentences = load_sentences(
            path=config.data_path,
            max_sentences=config.max_sentences,
            min_words=config.min_words,
            max_words=config.max_words,
            text_column=config.text_column,
        )

        items = make_prompts(
            sentences,
            prompt_ratio=config.prompt_ratio,
            min_prompt_words=config.min_prompt_words,
        )
        # Add provenance metadata (each text treated as a single sentence)
        for i, item in enumerate(items):
            item["text_id"] = i
            item["sentence_idx"] = 0

    logger.info("Prepared %d prompt-reference pairs", len(items))
    return items
