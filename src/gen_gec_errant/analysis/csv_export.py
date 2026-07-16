"""CSV export functions for pipeline results."""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _attr(obj, key, default=None):
    """Access a field from either a dataclass object or a plain dict."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _clean_for_tsv(text: str) -> str:
    """Clean text for TSV output."""
    if not isinstance(text, str):
        return str(text)
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = " ".join(text.split())
    return text.strip()


def build_csv_rows(
    items: List[dict],
    all_results: Dict[str, dict],
    model_names: List[str],
) -> List[dict]:
    """Build flat rows for CSV export."""
    n_sentences = len(items)
    rows = []

    for i in range(n_sentences):
        row = {
            "sentence_id": i,
            "text_id": items[i].get("text_id", i),
            "sentence_idx": items[i].get("sentence_idx", 0),
            "prompt": items[i]["prompt"],
            "reference_continuation": items[i]["reference"],
            "full_original": items[i]["full"],
        }

        for model_name in model_names:
            if model_name not in all_results:
                continue

            res = all_results[model_name]
            prefix = model_name.replace("-", "_")

            continuation = res["continuations"][i] if i < len(res["continuations"]) else ""
            full_text = res["full_texts"][i] if i < len(res["full_texts"]) else ""
            row[f"{prefix}__continuation"] = _clean_for_tsv(continuation)
            row[f"{prefix}__full_text"] = _clean_for_tsv(full_text)

            corr_cont = res.get("corrected_continuations", [])
            corr_full = res.get("corrected_full_texts", [])
            row[f"{prefix}__corrected_continuation"] = _clean_for_tsv(corr_cont[i]) if i < len(corr_cont) else ""
            row[f"{prefix}__corrected_full_text"] = _clean_for_tsv(corr_full[i]) if i < len(corr_full) else ""

            ppl = res["perplexities"][i] if i < len(res["perplexities"]) else ""
            row[f"{prefix}__perplexity"] = round(ppl, 4) if isinstance(ppl, float) else ppl

            annotations = res.get("annotations", [])
            if i < len(annotations):
                ann = annotations[i]
                num_errors = _attr(ann, "num_errors", 0)
                errors = _attr(ann, "errors", [])
                type_counts = _attr(ann, "error_type_counts", {})
                row[f"{prefix}__num_errors"] = num_errors

                error_types = [_attr(e, "error_type", _attr(e, "type", "")) for e in errors]
                row[f"{prefix}__error_types"] = ";".join(error_types) if error_types else ""

                type_counts_str = ";".join(f"{k}={v}" for k, v in sorted(type_counts.items()))
                row[f"{prefix}__error_type_counts"] = type_counts_str

                error_details = []
                for e in errors:
                    orig = _attr(e, "original_tokens", _attr(e, "orig_tokens", "")) or ""
                    corr = _attr(e, "corrected_tokens", _attr(e, "corr_tokens", "")) or ""
                    etype = _attr(e, "error_type", _attr(e, "type", ""))
                    error_details.append(f"{orig}->{corr}({etype})")
                row[f"{prefix}__error_details"] = "|".join(error_details) if error_details else ""

                row[f"{prefix}__has_errors"] = 1 if num_errors > 0 else 0
            else:
                row[f"{prefix}__num_errors"] = ""
                row[f"{prefix}__error_types"] = ""
                row[f"{prefix}__error_type_counts"] = ""
                row[f"{prefix}__error_details"] = ""
                row[f"{prefix}__has_errors"] = ""

            # Full-text annotation columns
            ft_annotations = res.get("full_text_annotations", [])
            if i < len(ft_annotations):
                ft_ann = ft_annotations[i]
                ft_num = _attr(ft_ann, "num_errors", 0)
                ft_errors = _attr(ft_ann, "errors", [])
                row[f"{prefix}__ft_num_errors"] = ft_num
                row[f"{prefix}__ft_prompt_errors"] = _attr(ft_ann, "prompt_error_count", 0)
                row[f"{prefix}__ft_generation_errors"] = _attr(ft_ann, "generation_error_count", 0)

                ft_details = []
                for e in ft_errors:
                    orig = _attr(e, "original_tokens", _attr(e, "orig_tokens", "")) or ""
                    corr = _attr(e, "corrected_tokens", _attr(e, "corr_tokens", "")) or ""
                    etype = _attr(e, "error_type", _attr(e, "type", ""))
                    region = _attr(e, "region", "") or ""
                    tag = f"[{region}]" if region else ""
                    ft_details.append(f"{tag}{orig}->{corr}({etype})")
                row[f"{prefix}__ft_error_details"] = "|".join(ft_details) if ft_details else ""
            else:
                row[f"{prefix}__ft_num_errors"] = ""
                row[f"{prefix}__ft_prompt_errors"] = ""
                row[f"{prefix}__ft_generation_errors"] = ""
                row[f"{prefix}__ft_error_details"] = ""

        rows.append(row)

    return rows


def export_csv(
    items: List[dict],
    all_results: Dict[str, dict],
    model_names: List[str],
    output_path: str,
) -> str:
    """Export the full pipeline results to a single TSV file."""
    output_path = Path(output_path).with_suffix(".tsv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = build_csv_rows(items, all_results, model_names)

    if not rows:
        logger.warning("No rows to export")
        return str(output_path)

    fieldnames = list(rows[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t",
                                quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Exported %d rows x %d columns to %s", len(rows), len(fieldnames), output_path)
    return str(output_path)


def export_errors_long_format(
    all_results: Dict[str, dict],
    model_names: List[str],
    output_path: str,
    items: Optional[List[dict]] = None,
) -> str:
    """Export errors in long format (one row per error)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _item_meta(sent_id):
        if items and sent_id < len(items):
            return items[sent_id].get("text_id", sent_id), items[sent_id].get("sentence_idx", 0)
        return sent_id, 0

    rows = []
    for model_name in model_names:
        if model_name not in all_results:
            continue

        # Continuation annotations
        annotations = all_results[model_name].get("annotations", [])
        for sent_id, ann in enumerate(annotations):
            text_id, sentence_idx = _item_meta(sent_id)
            errors = _attr(ann, "errors", [])
            for err in errors:
                etype = _attr(err, "error_type", _attr(err, "type", ""))
                parts = etype.split(":", 1)
                category = parts[0] if parts else ""
                subcategory = parts[1] if len(parts) > 1 else ""

                rows.append({
                    "sentence_id": sent_id,
                    "text_id": text_id,
                    "sentence_idx": sentence_idx,
                    "model": model_name,
                    "source": "continuation",
                    "original_text": _clean_for_tsv(_attr(ann, "original", "")),
                    "corrected_text": _clean_for_tsv(_attr(ann, "corrected", "")),
                    "error_original_tokens": _attr(err, "original_tokens", _attr(err, "orig_tokens", "")),
                    "error_corrected_tokens": _attr(err, "corrected_tokens", _attr(err, "corr_tokens", "")),
                    "error_type": etype,
                    "error_operation": category,
                    "error_subcategory": subcategory,
                    "start_offset": _attr(err, "start_offset", 0),
                    "end_offset": _attr(err, "end_offset", 0),
                    "char_start": _attr(err, "char_start", ""),
                    "char_end": _attr(err, "char_end", ""),
                    "region": "",
                })

        # Full-text annotations
        ft_annotations = all_results[model_name].get("full_text_annotations", [])
        for sent_id, ann in enumerate(ft_annotations):
            text_id, sentence_idx = _item_meta(sent_id)
            errors = _attr(ann, "errors", [])
            for err in errors:
                etype = _attr(err, "error_type", _attr(err, "type", ""))
                parts = etype.split(":", 1)
                category = parts[0] if parts else ""
                subcategory = parts[1] if len(parts) > 1 else ""

                rows.append({
                    "sentence_id": sent_id,
                    "text_id": text_id,
                    "sentence_idx": sentence_idx,
                    "model": model_name,
                    "source": "full_text",
                    "original_text": _clean_for_tsv(_attr(ann, "original", "")),
                    "corrected_text": _clean_for_tsv(_attr(ann, "corrected", "")),
                    "error_original_tokens": _attr(err, "original_tokens", _attr(err, "orig_tokens", "")),
                    "error_corrected_tokens": _attr(err, "corrected_tokens", _attr(err, "corr_tokens", "")),
                    "error_type": etype,
                    "error_operation": category,
                    "error_subcategory": subcategory,
                    "start_offset": _attr(err, "start_offset", 0),
                    "end_offset": _attr(err, "end_offset", 0),
                    "char_start": _attr(err, "char_start", ""),
                    "char_end": _attr(err, "char_end", ""),
                    "region": _attr(err, "region", ""),
                })

    fieldnames = [
        "sentence_id", "text_id", "sentence_idx",
        "model", "source", "original_text", "corrected_text",
        "error_original_tokens", "error_corrected_tokens", "error_type",
        "error_operation", "error_subcategory", "start_offset", "end_offset",
        "char_start", "char_end", "region",
    ]

    if not rows:
        logger.warning("No errors to export in long format")

    output_path = Path(output_path).with_suffix(".tsv")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t",
                                quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Exported %d error rows (long format) to %s", len(rows), output_path)
    return str(output_path)
