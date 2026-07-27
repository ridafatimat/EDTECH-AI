from __future__ import annotations

import re

from app.schemas.transcript import (
    PreprocessingResult,
    PreprocessingStats,
)


# =========================================================
# PATTERNS
# =========================================================

# Examples:
# 00:12
# 01:05:31
# [00:12]
# (00:12)
TIMESTAMP_PATTERN = re.compile(
    r"""
    ^\s*
    [\[\(]?
    \d{1,2}:\d{2}
    (?::\d{2})?
    [\]\)]?
    \s*
    """,
    re.VERBOSE,
)


# Example:
# 00:10 --> 00:15
TIMESTAMP_RANGE_PATTERN = re.compile(
    r"""
    ^\s*
    \d{1,2}:\d{2}(?::\d{2})?
    \s*-->\s*
    \d{1,2}:\d{2}(?::\d{2})?
    \s*
    """,
    re.VERBOSE,
)


# Common transcript speaker labels.
#
# Examples:
# Teacher:
# Student:
# Speaker 1:
# Instructor:
SPEAKER_PATTERN = re.compile(
    r"""
    ^\s*
    (?:
        teacher
        | instructor
        | lecturer
        | tutor
        | student
        | speaker\s*\d*
    )
    \s*:\s*
    """,
    re.IGNORECASE | re.VERBOSE,
)


# Combined speaker + timestamp prefix.
#
# Examples:
# Teacher - 16:00:01:
# Student - 16:00:05:
# Speaker 1 - 16:00:08:
# Teacher – 16:00:
#
# This is handled separately so that both
# timestamps_removed and speaker_labels_removed
# can be counted correctly.
SPEAKER_TIMESTAMP_PATTERN = re.compile(
    r"""
    ^\s*
    (?:
        teacher
        | instructor
        | lecturer
        | tutor
        | student
        | speaker\s*\d*
    )
    \s*
    [-–—]
    \s*
    [\[\(]?
    \d{1,2}:\d{2}
    (?::\d{2})?
    [\]\)]?
    \s*
    :\s*
    """,
    re.IGNORECASE | re.VERBOSE,
)


# Deterministic non-content transcript artefacts.
#
# Only known non-verbal / system-event markers are removed.
# We do NOT use a broad r"\[.*?\]" rule because brackets may
# contain useful educational content in future transcripts.
ARTEFACT_PATTERN = re.compile(
    r"""
    [\[\(]
        \s*
        (?:
            inaudible
            | unintelligible
            | noise
            | background\s+noise
            | keyboard\s+noise
            | typing
            | chair\s+noise
            | notification\s+sound
            | notification
            | audio\s+glitch
            | microphone\s+noise
            | mic\s+noise
            | music
            | silence
            | laughter
            | laughing
            | crosstalk
            | cough
            | coughing
            | screen\s+sharing\s+(?:started|stopped)
            | connection\s+(?:lost|restored)
            | internet\s+(?:lost|disconnected|reconnected)
        )
        \s*
    [\]\)]
    """,
    re.IGNORECASE | re.VERBOSE,
)


# Missing/uncertain speech markers.
UNCERTAINTY_PATTERN = re.compile(
    r"""
    \[unclear\]
    |
    \[unknown\]
    |
    \?\?\?
    |
    <unk>
    """,
    re.IGNORECASE | re.VERBOSE,
)


# Conservative fillers only.
FILLER_PATTERN = re.compile(
    r"\b(?:um+|uh+|erm+|hmm+)\b[,\s]*",
    re.IGNORECASE,
)


# Huge malformed ASR token.
LONG_TOKEN_PATTERN = re.compile(
    r"\b[A-Za-z]{30,}\b"
)


# =========================================================
# BASIC CLEANING
# =========================================================

def remove_speaker_timestamp_prefix(
    line: str,
) -> tuple[str, bool]:
    """
    Remove a combined speaker + timestamp prefix.

    Example:
        Teacher - 16:00:01: Hello

    becomes:
        Hello

    A True return value means BOTH a speaker label and
    a timestamp were removed.
    """

    cleaned = SPEAKER_TIMESTAMP_PATTERN.sub(
        "",
        line,
    )

    return cleaned, cleaned != line


def remove_timestamp(
    line: str,
) -> tuple[str, bool]:
    """
    Remove timestamps occurring at the beginning of a line.
    """

    cleaned = TIMESTAMP_RANGE_PATTERN.sub(
        "",
        line,
    )

    if cleaned != line:
        return cleaned, True

    cleaned = TIMESTAMP_PATTERN.sub(
        "",
        line,
    )

    return cleaned, cleaned != line


def remove_speaker_label(
    line: str,
) -> tuple[str, bool]:
    """
    Remove common transcript speaker labels.
    """

    cleaned = SPEAKER_PATTERN.sub(
        "",
        line,
    )

    return cleaned, cleaned != line


def remove_artefacts(
    text: str,
) -> tuple[str, int]:
    """
    Remove known non-verbal transcript artefacts.

    Examples:
        [background noise]
        [keyboard noise]
        [audio glitch]
        [screen sharing started]
        [connection lost]
        [notification sound]
    """

    return ARTEFACT_PATTERN.subn(
        "",
        text,
    )


def remove_uncertainty_markers(
    text: str,
) -> tuple[str, int]:
    """
    Remove uncertainty markers without guessing
    the missing speech.
    """

    return UNCERTAINTY_PATTERN.subn(
        "",
        text,
    )


def remove_fillers(
    text: str,
) -> tuple[str, int]:
    """
    Remove only very safe filler words.
    """

    return FILLER_PATTERN.subn(
        "",
        text,
    )


# =========================================================
# REPETITION CLEANING
# =========================================================

def compress_repeated_words(
    text: str,
) -> tuple[str, int]:
    """
    Compress 3 or more immediately repeated identical words.

    Example:

        you you you you you

    becomes:

        you

    Two-word repetitions are preserved because they may be
    natural speech.
    """

    removed_count = 0

    pattern = re.compile(
        r"\b([A-Za-z]+)"
        r"(?:\s+\1\b){2,}",
        re.IGNORECASE,
    )

    def replace(
        match: re.Match,
    ) -> str:

        nonlocal removed_count

        matched_text = match.group(0)

        words = matched_text.split()

        removed_count += (
            len(words) - 1
        )

        return match.group(1)

    cleaned = pattern.sub(
        replace,
        text,
    )

    return cleaned, removed_count


def compress_repeated_sentences(
    text: str,
) -> tuple[str, int]:
    """
    Remove consecutive duplicate sentences.

    Example:

        Okay. Okay. Okay.

    becomes:

        Okay.

    Also handles long repeated ASR sentences such as:

        I'm going to start with the first one.
        I'm going to start with the first one.
        ...
    """

    removed_count = 0
    cleaned_lines: list[str] = []

    for line in text.splitlines():

        stripped_line = line.strip()

        if not stripped_line:
            cleaned_lines.append("")
            continue

        sentences = re.split(
            r"(?<=[.!?])\s+",
            stripped_line,
        )

        kept_sentences: list[str] = []

        previous_normalized: str | None = None

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            normalized = re.sub(
                r"\s+",
                " ",
                sentence,
            ).strip().lower()

            if (
                previous_normalized is not None
                and normalized == previous_normalized
            ):
                removed_count += 1
                continue

            kept_sentences.append(
                sentence
            )

            previous_normalized = normalized

        cleaned_lines.append(
            " ".join(kept_sentences)
        )

    return (
        "\n".join(cleaned_lines),
        removed_count,
    )


# =========================================================
# WHITESPACE
# =========================================================

def normalize_whitespace(
    text: str,
) -> str:
    """
    Normalize spacing without semantically rewriting
    the transcript.
    """

    text = text.replace(
        "\r\n",
        "\n",
    )

    text = text.replace(
        "\r",
        "\n",
    )

    text = text.replace(
        "\u00a0",
        " ",
    )

    cleaned_lines: list[str] = []

    for line in text.splitlines():

        line = re.sub(
            r"[ \t]+",
            " ",
            line,
        )

        cleaned_lines.append(
            line.strip()
        )

    text = "\n".join(
        cleaned_lines
    )

    # Remove excessive blank lines.
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    # Remove accidental spaces before punctuation.
    text = re.sub(
        r"\s+([,.!?;:])",
        r"\1",
        text,
    )

    return text.strip()


# =========================================================
# QUALITY WARNINGS
# =========================================================

def collect_warnings(
    cleaned_text: str,
) -> list[str]:
    """
    Detect remaining transcription problems.

    These warnings do NOT trigger GPT here.
    Later modules can decide whether a relevant chunk
    requires fallback handling.
    """

    warnings: list[str] = []

    malformed_tokens = LONG_TOKEN_PATTERN.findall(
        cleaned_text
    )

    if malformed_tokens:

        warnings.append(
            "Possible malformed ASR token remains."
        )

    if "�" in cleaned_text:

        warnings.append(
            "Possible corrupted character remains."
        )

    return warnings


# =========================================================
# MAIN MODULE 1 PREPROCESSOR
# =========================================================

def preprocess_transcript(
    raw_text: str,
) -> PreprocessingResult:
    """
    Lightweight deterministic transcript preprocessing.

    This module deliberately avoids semantic rewriting.

    Steps:
        1. Validate input
        2. Remove combined speaker + timestamp prefixes
        3. Remove standalone timestamps
        4. Remove standalone speaker labels
        5. Remove obvious non-verbal artefacts
        6. Remove uncertainty markers
        7. Remove safe fillers
        8. Compress repeated words
        9. Compress repeated sentences
        10. Normalize whitespace
        11. Return cleaned transcript for semantic chunking
    """

    if not raw_text or not raw_text.strip():

        raise ValueError(
            "Transcript cannot be empty."
        )

    original_characters = len(
        raw_text
    )

    timestamps_removed = 0
    speaker_labels_removed = 0

    processed_lines: list[str] = []

    # =====================================================
    # LINE-LEVEL CLEANING
    # =====================================================

    for line in raw_text.splitlines():

        # First handle formats such as:
        # Teacher - 16:00:01:
        # Student - 16:00:05:
        (
            line,
            combined_prefix_removed,
        ) = remove_speaker_timestamp_prefix(
            line
        )

        if combined_prefix_removed:
            timestamps_removed += 1
            speaker_labels_removed += 1

        else:
            # Existing formats such as:
            # 00:10 Teacher:
            # [00:10] Speaker 1:
            line, timestamp_removed = (
                remove_timestamp(
                    line
                )
            )

            if timestamp_removed:
                timestamps_removed += 1

            line, speaker_removed = (
                remove_speaker_label(
                    line
                )
            )

            if speaker_removed:
                speaker_labels_removed += 1

        processed_lines.append(
            line
        )

    text = "\n".join(
        processed_lines
    )

    # =====================================================
    # DETERMINISTIC CLEANING
    # =====================================================

    text, artefacts_removed = remove_artefacts(
        text
    )

    (
        text,
        uncertainty_markers_removed,
    ) = remove_uncertainty_markers(
        text
    )

    text, fillers_removed = remove_fillers(
        text
    )

    (
        text,
        repeated_words_removed,
    ) = compress_repeated_words(
        text
    )

    (
        text,
        repeated_sentences_removed,
    ) = compress_repeated_sentences(
        text
    )

    text = normalize_whitespace(
        text
    )

    # =====================================================
    # WARNINGS
    # =====================================================

    warnings = collect_warnings(
        text
    )

    # =====================================================
    # STATS
    # =====================================================

    stats = PreprocessingStats(
        original_characters=original_characters,
        cleaned_characters=len(text),

        timestamps_removed=(
            timestamps_removed
        ),

        speaker_labels_removed=(
            speaker_labels_removed
        ),

        fillers_removed=(
            fillers_removed
        ),

        artefacts_removed=(
            artefacts_removed
        ),

        uncertainty_markers_removed=(
            uncertainty_markers_removed
        ),

        repeated_words_removed=(
            repeated_words_removed
        ),

        repeated_sentences_removed=(
            repeated_sentences_removed
        ),
    )

    return PreprocessingResult(
        cleaned_text=text,
        warnings=warnings,
        stats=stats,
    )