"""Answer quality evaluation module for the academic RAG system.

This module provides comprehensive answer quality evaluation and validation:
- Faithfulness: Are claims supported by retrieved evidence?
- Coverage: Did the answer cover key papers/angles?
- Utility: Is the answer publication-ready?
- Validation: Citation verification and hallucination detection
"""

from .answer_quality import (
    AnswerQualityEvaluator,
    QualityMetrics,
)

from .answer_validation import (
    AnswerValidator,
    ValidationResult,
)

__all__ = [
    "AnswerQualityEvaluator",
    "QualityMetrics",
    "AnswerValidator",
    "ValidationResult",
]

