"""Preprocessing runner: convert EFCAMDAT essays to sentence-level data."""

import csv
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from gen_gec_errant.preprocessing.config import PreprocessingConfig

logger = logging.getLogger(__name__)


def detect_columns(header: List[str]) -> Dict[str, int]:
    """Auto-detect which columns contain what, based on header names."""
    header_lower = [h.strip().lower() for h in header]
    mapping: Dict[str, int] = {}

    for role, candidates in [
        ("original_text", ["text", "original_text", "writing", "essay", "learner_text"]),
        ("corrected_text", ["corrected", "cleaned_text", "corrected_text"]),
        ("cefr_level", ["cefr", "cefr_level", "level", "proficiency"]),
        ("l1", ["l1", "l1_language", "language", "native_language", "mother_tongue"]),
        ("topic", ["topic", "task", "prompt", "writing_topic"]),
    ]:
        for cand in candidates:
            for i, h in enumerate(header_lower):
                if cand == h or cand in h:
                    if role not in mapping:
                        mapping[role] = i
                    break

    return mapping


def detect_columns_by_position(header: List[str], sample_rows: List[list]) -> Dict[str, int]:
    """Fallback: detect text columns by content heuristics."""
    if not sample_rows:
        return {}

    avg_lengths = []
    for col_idx in range(len(header)):
        lengths = [len(str(row[col_idx])) for row in sample_rows if col_idx < len(row)]
        avg_lengths.append(sum(lengths) / max(len(lengths), 1))

    sorted_cols = sorted(enumerate(avg_lengths), key=lambda x: -x[1])

    mapping: Dict[str, int] = {}
    text_cols_found = 0
    for col_idx, avg_len in sorted_cols:
        if avg_len > 50 and text_cols_found < 2:
            if text_cols_found == 0:
                mapping["original_text"] = col_idx
            else:
                mapping["corrected_text"] = col_idx
            text_cols_found += 1

    cefr_pattern = re.compile(r'^[ABC][12]$')
    for col_idx in range(len(header)):
        vals = {str(row[col_idx]).strip() for row in sample_rows if col_idx < len(row)}
        if any(cefr_pattern.match(v) for v in vals):
            mapping["cefr_level"] = col_idx
            break

    lang_names = {"arabic", "mandarin", "french", "portuguese", "italian", "spanish",
                  "german", "japanese", "korean", "russian", "turkish", "thai", "hindi"}
    for col_idx in range(len(header)):
        vals = {str(row[col_idx]).strip().lower() for row in sample_rows if col_idx < len(row)}
        if vals & lang_names:
            mapping["l1"] = col_idx
            break

    return mapping


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using spaCy or regex fallback."""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)

    try:
        import spacy
        if not hasattr(split_into_sentences, '_nlp'):
            split_into_sentences._nlp = spacy.load("en_core_web_sm")
        doc = split_into_sentences._nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    except (ImportError, OSError):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


def clean_text(text: str) -> str:
    """Basic text cleaning."""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text.strip())
    text = text.replace('\n', ' ').replace('\r', ' ')
    return text


def run_preprocessing(config: PreprocessingConfig) -> Path:
    """
    Run EFCAMDAT preprocessing.

    Returns:
        Path to the output sentence-level CSV.
    """
    input_path = Path(config.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    cefr_filter = None
    if config.cefr_filter:
        cefr_filter = set(config.cefr_filter.upper().split(","))

    l1_filter = None
    if config.l1_filter:
        l1_filter = set(x.strip().lower() for x in config.l1_filter.split(","))

    logger.info("Reading %s...", input_path)

    rows_raw: list[list[str]] = []
    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            rows_raw.append(row)

    logger.info("Read %d essays", len(rows_raw))

    # Detect columns
    col_map = detect_columns(header)
    if not col_map.get("original_text"):
        fallback = detect_columns_by_position(header, rows_raw[:100])
        for k, v in fallback.items():
            if k not in col_map:
                col_map[k] = v

    if config.text_col:
        col_map["original_text"] = (
            int(config.text_col) if config.text_col.isdigit()
            else header.index(config.text_col)
        )
    if config.corrected_col:
        col_map["corrected_text"] = (
            int(config.corrected_col) if config.corrected_col.isdigit()
            else header.index(config.corrected_col)
        )

    if "original_text" not in col_map:
        raise ValueError(
            f"Could not detect text column. Available: {list(enumerate(header))}"
        )

    text_idx = col_map["original_text"]
    cefr_idx = col_map.get("cefr_level")
    l1_idx = col_map.get("l1")
    topic_idx = col_map.get("topic")

    # Process essays into sentences
    output_rows = []
    essay_count = 0

    for row in rows_raw:
        if config.max_essays and essay_count >= config.max_essays:
            break

        if text_idx >= len(row):
            continue

        cefr = row[cefr_idx].strip() if (cefr_idx is not None and cefr_idx < len(row)) else ""
        l1 = row[l1_idx].strip() if (l1_idx is not None and l1_idx < len(row)) else ""
        topic = row[topic_idx].strip() if (topic_idx is not None and topic_idx < len(row)) else ""

        if cefr_filter and cefr.upper() not in cefr_filter:
            continue
        if l1_filter and l1.lower() not in l1_filter:
            continue

        original_text = clean_text(row[text_idx])
        if not original_text:
            continue

        orig_sentences = split_into_sentences(original_text)

        for sent_idx, sentence in enumerate(orig_sentences):
            words = sentence.split()
            if len(words) < config.min_words or len(words) > config.max_words:
                continue

            output_rows.append({
                "essay_id": essay_count,
                "sentence_idx": sent_idx,
                "sentence": sentence,
                "cefr_level": cefr,
                "l1_language": l1,
                "topic": topic,
                "word_count": len(words),
            })

        essay_count += 1

    logger.info("Processed %d essays -> %d sentences", essay_count, len(output_rows))

    # Write output
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["essay_id", "sentence_idx", "sentence", "cefr_level",
                  "l1_language", "topic", "word_count"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    logger.info("Saved to %s", output_path)
    return output_path
