"""Shared data types for the pipeline."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ErrorAnnotation:
    """A single error found by ERRANT."""
    original_tokens: str
    corrected_tokens: str
    error_type: str       # e.g. "M:DET", "R:VERB:SVA"
    start_offset: int
    end_offset: int
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    region: Optional[str] = None  # "prompt" or "generation"


@dataclass
class SentenceAnnotation:
    """All errors found in one sentence."""
    original: str
    corrected: str
    errors: List[ErrorAnnotation] = field(default_factory=list)
    num_errors: int = 0
    error_type_counts: Dict[str, int] = field(default_factory=dict)
    prompt_error_count: int = 0
    generation_error_count: int = 0
    prompt_error_type_counts: Dict[str, int] = field(default_factory=dict)
    generation_error_type_counts: Dict[str, int] = field(default_factory=dict)
