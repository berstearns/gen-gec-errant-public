"""Annotation runner: ERRANT error annotation on original vs corrected pairs."""

import logging
from typing import Dict, List

from gen_gec_errant._types import ErrorAnnotation, SentenceAnnotation
from gen_gec_errant.annotation.config import AnnotationConfig

logger = logging.getLogger(__name__)


class ERRANTAnnotator:
    """Wrapper around ERRANT for batch error annotation."""

    def __init__(self, lang: str = "en"):
        try:
            import errant
        except ImportError:
            raise ImportError(
                "ERRANT is required. Install with:\n"
                "  pip install errant\n"
                "  python -m spacy download en_core_web_sm"
            )
        self.annotator = errant.load(lang)
        logger.info("ERRANT annotator loaded")

    def annotate_pair(self, original: str, corrected: str) -> SentenceAnnotation:
        """Annotate errors between an original and corrected sentence."""
        orig_parsed = self.annotator.parse(original)
        corr_parsed = self.annotator.parse(corrected)
        edits = self.annotator.annotate(orig_parsed, corr_parsed)

        errors = []
        type_counts: Dict[str, int] = {}

        for edit in edits:
            if edit.type == "noop":
                continue

            # Convert token indices to character offsets
            if edit.o_start < len(orig_parsed) and edit.o_end > edit.o_start:
                char_start = orig_parsed[edit.o_start].idx
                last_tok = orig_parsed[edit.o_end - 1]
                char_end = last_tok.idx + len(last_tok.text)
            elif edit.o_start < len(orig_parsed):
                # Insertion (o_start == o_end)
                char_start = orig_parsed[edit.o_start].idx
                char_end = char_start
            else:
                # Insertion at end of doc
                char_start = len(original)
                char_end = char_start

            err = ErrorAnnotation(
                original_tokens=edit.o_str,
                corrected_tokens=edit.c_str,
                error_type=edit.type,
                start_offset=edit.o_start,
                end_offset=edit.o_end,
                char_start=char_start,
                char_end=char_end,
            )
            errors.append(err)
            type_counts[edit.type] = type_counts.get(edit.type, 0) + 1

        return SentenceAnnotation(
            original=original,
            corrected=corrected,
            errors=errors,
            num_errors=len(errors),
            error_type_counts=type_counts,
        )

    def annotate_batch(
        self,
        originals: List[str],
        correcteds: List[str],
    ) -> List[SentenceAnnotation]:
        """Annotate a batch of sentence pairs."""
        assert len(originals) == len(correcteds), "Mismatched lengths"

        annotations = []
        for orig, corr in zip(originals, correcteds):
            try:
                ann = self.annotate_pair(orig, corr)
            except Exception as e:
                logger.warning("ERRANT annotation failed for: '%s...' -> %s", orig[:50], e)
                ann = SentenceAnnotation(original=orig, corrected=corr)
            annotations.append(ann)

        return annotations


def summarize_errors(annotations: List[SentenceAnnotation]) -> Dict:
    """Aggregate error statistics across all annotated sentences."""
    total_errors = 0
    total_sentences = len(annotations)
    sentences_with_errors = 0
    global_type_counts: Dict[str, int] = {}
    errors_per_sentence = []

    for ann in annotations:
        total_errors += ann.num_errors
        errors_per_sentence.append(ann.num_errors)
        if ann.num_errors > 0:
            sentences_with_errors += 1
        for etype, count in ann.error_type_counts.items():
            global_type_counts[etype] = global_type_counts.get(etype, 0) + count

    avg_errors = total_errors / max(total_sentences, 1)
    sorted_types = sorted(global_type_counts.items(), key=lambda x: -x[1])

    return {
        "total_sentences": total_sentences,
        "total_errors": total_errors,
        "sentences_with_errors": sentences_with_errors,
        "error_rate": sentences_with_errors / max(total_sentences, 1),
        "avg_errors_per_sentence": avg_errors,
        "error_type_counts": dict(sorted_types),
        "errors_per_sentence": errors_per_sentence,
        "top_10_error_types": sorted_types[:10],
    }


def classify_errors_by_region(
    annotations: List[SentenceAnnotation],
    prompt_boundaries: List[int],
) -> None:
    """Classify each error as 'prompt' or 'generation' based on char offset.

    Mutates annotations in place: sets region on each error, and populates
    prompt/generation error counts and type counts on each SentenceAnnotation.
    """
    for ann, boundary in zip(annotations, prompt_boundaries):
        prompt_count = 0
        gen_count = 0
        prompt_types: Dict[str, int] = {}
        gen_types: Dict[str, int] = {}

        for err in ann.errors:
            if err.char_start is not None and err.char_start < boundary:
                err.region = "prompt"
                prompt_count += 1
                prompt_types[err.error_type] = prompt_types.get(err.error_type, 0) + 1
            else:
                err.region = "generation"
                gen_count += 1
                gen_types[err.error_type] = gen_types.get(err.error_type, 0) + 1

        ann.prompt_error_count = prompt_count
        ann.generation_error_count = gen_count
        ann.prompt_error_type_counts = prompt_types
        ann.generation_error_type_counts = gen_types


def summarize_errors_by_region(annotations: List[SentenceAnnotation]) -> Dict:
    """Aggregate prompt vs generation error totals and type distributions."""
    prompt_total = 0
    gen_total = 0
    prompt_type_counts: Dict[str, int] = {}
    gen_type_counts: Dict[str, int] = {}

    for ann in annotations:
        prompt_total += ann.prompt_error_count
        gen_total += ann.generation_error_count
        for etype, count in ann.prompt_error_type_counts.items():
            prompt_type_counts[etype] = prompt_type_counts.get(etype, 0) + count
        for etype, count in ann.generation_error_type_counts.items():
            gen_type_counts[etype] = gen_type_counts.get(etype, 0) + count

    return {
        "prompt_total_errors": prompt_total,
        "generation_total_errors": gen_total,
        "prompt_error_type_counts": dict(sorted(prompt_type_counts.items(), key=lambda x: -x[1])),
        "generation_error_type_counts": dict(sorted(gen_type_counts.items(), key=lambda x: -x[1])),
    }


def run_annotation(config: AnnotationConfig, gec_results: dict) -> dict:
    """
    Run ERRANT annotation on GEC results.

    Args:
        config: AnnotationConfig
        gec_results: Dict with continuations and corrected_continuations

    Returns:
        Updated dict with annotations and error_summary added.
    """
    annotator = ERRANTAnnotator(lang=config.lang)

    logger.info("Annotating errors...")
    annotations = annotator.annotate_batch(
        gec_results["continuations"],
        gec_results["corrected_continuations"],
    )
    gec_results["annotations"] = annotations

    summary = summarize_errors(annotations)
    gec_results["error_summary"] = summary

    logger.info(
        "%d errors in %d sentences (rate: %.3f)",
        summary["total_errors"],
        summary["total_sentences"],
        summary["error_rate"],
    )

    # Full-text annotation
    if "full_texts" in gec_results and "corrected_full_texts" in gec_results:
        full_text_annotations = annotator.annotate_batch(
            gec_results["full_texts"], gec_results["corrected_full_texts"])
        gec_results["full_text_annotations"] = full_text_annotations
        gec_results["full_text_error_summary"] = summarize_errors(full_text_annotations)

    return gec_results
