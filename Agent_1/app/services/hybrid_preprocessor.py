from __future__ import annotations

import re
from collections import Counter

from app.schemas.transcript import (
    HybridPreprocessingResult,
)
from app.services.transcript_llm_refiner import (
    refine_transcript_with_llm,
)
from app.services.transcript_preprocessor import (
    preprocess_transcript,
)


# =========================================================
# GENERIC TECHNICAL SAFETY PATTERNS
# =========================================================

# Examples:
# array[mid]
# numbers[i]
# values[index + 1]
ARRAY_REFERENCE_PATTERN = re.compile(
    r"\b[A-Za-z_]\w*\[[^\]]+\]"
)


# Important programming operators.
PROTECTED_OPERATORS = [
    "==",
    "!=",
    "<=",
    ">=",
    "+=",
    "-=",
    "*=",
    "/=",
    "//",
]


# =========================================================
# TECHNICAL CONTENT EXTRACTION
# =========================================================

def extract_array_references(
    text: str,
) -> list[str]:
    """
    Extract array/list indexing expressions.
    """

    return ARRAY_REFERENCE_PATTERN.findall(
        text
    )


def count_protected_operators(
    text: str,
) -> dict[str, int]:
    """
    Count important programming operators.

    The goal is to make sure GPT-OSS does not silently
    change technical logic.
    """

    return {
        operator: text.count(operator)
        for operator in PROTECTED_OPERATORS
    }


# =========================================================
# GENERIC SAFETY VALIDATOR
# =========================================================

def validate_llm_output(
    original_text: str,
    refined_text: str,
) -> tuple[bool, list[str]]:
    """
    Generic deterministic validation of GPT-OSS output.

    This does not hardcode specific transcript mistakes.

    It protects broad invariants:

        1. Output cannot be empty.
        2. GPT must not delete most of the transcript.
        3. GPT must not massively expand the transcript.
        4. Array/index expressions must survive.
        5. Important programming operators must survive.

    Irrelevant classroom chatter is allowed to disappear.
    """

    problems: list[str] = []

    original_text = original_text.strip()
    refined_text = refined_text.strip()

    # =====================================================
    # CHECK 1: EMPTY OUTPUT
    # =====================================================

    if not refined_text:

        problems.append(
            "LLM returned an empty transcript."
        )

        return False, problems

    # =====================================================
    # CHECK 2: EXTREME CONTENT DELETION / ADDITION
    # =====================================================

    original_length = len(
        original_text
    )

    refined_length = len(
        refined_text
    )

    if original_length > 0:

        length_ratio = (
            refined_length
            / original_length
        )

        # Classroom chatter and bad fragments may legitimately
        # be removed, so we allow a reasonable reduction.
        #
        # But deleting more than half of the transcript is
        # suspicious.
        if length_ratio < 0.50:

            problems.append(
                "LLM removed an unusually large amount "
                "of transcript content."
            )

        # Cleaning should not substantially expand the lesson.
        if length_ratio > 1.20:

            problems.append(
                "LLM added an unusually large amount "
                "of transcript content."
            )

    # =====================================================
    # CHECK 3: ARRAY / INDEX EXPRESSIONS
    # =====================================================

    original_arrays = Counter(
        extract_array_references(
            original_text
        )
    )

    refined_arrays = Counter(
        extract_array_references(
            refined_text
        )
    )

    for reference, original_count in (
        original_arrays.items()
    ):

        if refined_arrays[reference] < original_count:

            problems.append(
                "LLM modified or removed technical "
                f"expression: {reference}"
            )

    # =====================================================
    # CHECK 4: PROGRAMMING OPERATORS
    # =====================================================

    original_operators = (
        count_protected_operators(
            original_text
        )
    )

    refined_operators = (
        count_protected_operators(
            refined_text
        )
    )

    for operator, original_count in (
        original_operators.items()
    ):

        if (
            refined_operators[operator]
            != original_count
        ):

            problems.append(
                "LLM changed programming operator "
                f"usage: {operator}"
            )

    # =====================================================
    # FINAL RESULT
    # =====================================================

    is_safe = len(problems) == 0

    return (
        is_safe,
        problems,
    )


# =========================================================
# COMPLETE MODULE 1 PIPELINE
# =========================================================

def preprocess_transcript_hybrid(
    raw_text: str,
) -> HybridPreprocessingResult:
    """
    Complete Module 1 preprocessing.

    Flow:

        Raw transcript
            ↓
        Python/Regex mechanical cleaning
            ↓
        GPT-OSS contextual cleaning
            ↓
        Generic deterministic safety validation
            ↓
        Safe GPT result → accept
        Unsafe GPT result → use deterministic result
            ↓
        Final cleaned transcript
    """

    # =====================================================
    # STEP 1:
    # DETERMINISTIC CLEANING
    # =====================================================

    rule_result = preprocess_transcript(
        raw_text
    )

    # =====================================================
    # STEP 2:
    # CONTEXTUAL GPT-OSS CLEANING
    # =====================================================

    try:

        llm_result = refine_transcript_with_llm(
            cleaned_text=rule_result.cleaned_text,
            fallback_reasons=(
                rule_result.fallback_reasons
            ),
        )

    except Exception as error:

        raise RuntimeError(
            "GPT-OSS contextual transcript cleaning "
            f"failed: {error}"
        ) from error

    # =====================================================
    # STEP 3:
    # SAFETY VALIDATION
    # =====================================================

    is_safe, validation_problems = (
        validate_llm_output(
            original_text=(
                rule_result.cleaned_text
            ),
            refined_text=(
                llm_result.refined_text
            ),
        )
    )

    # =====================================================
    # STEP 4A:
    # ACCEPT GPT OUTPUT
    # =====================================================

    if is_safe:

        final_text = (
            llm_result.refined_text.strip()
        )

        llm_accepted = True

        llm_changes = (
            llm_result.changes_made
        )

    # =====================================================
    # STEP 4B:
    # REJECT GPT OUTPUT
    # =====================================================

    else:

        final_text = (
            rule_result.cleaned_text
        )

        llm_accepted = False

        llm_changes = [
            (
                "GPT-OSS refinement was rejected "
                "by generic safety validation."
            )
        ]

        llm_changes.extend(
            validation_problems
        )

    # =====================================================
    # STEP 5:
    # FINAL STATS
    # =====================================================

    final_stats = (
        rule_result.stats.model_copy(
            update={
                "cleaned_characters": len(
                    final_text
                )
            }
        )
    )

    # =====================================================
    # STEP 6:
    # FINAL RESULT
    # =====================================================

    return HybridPreprocessingResult(
        cleaned_text=final_text,

        llm_used=True,

        llm_accepted=llm_accepted,

        llm_changes=llm_changes,

        stats=final_stats,
    )