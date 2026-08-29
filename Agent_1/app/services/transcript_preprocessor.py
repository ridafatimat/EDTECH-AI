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
            | notification\s+(?:sound|noise)
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
    r"""
    (?<![\w-])
    (?:um+|uh+|erm+|hmm+)
    (?![\w-])
    (?:[ \t]*,[ \t]*)?
    """,
    re.IGNORECASE | re.VERBOSE,
)


# Huge malformed ASR token.
LONG_TOKEN_PATTERN = re.compile(
    r"\b[A-Za-z]{30,}\b"
)




# =========================================================
# DOCUMENT METADATA
# =========================================================

# Metadata removal is deliberately conservative. A line is removed only when
# it has strong document-level signals. Ordinary lesson introductions such as
# "Today we are learning linear search" do not match these patterns.
PAGE_METADATA_PATTERN = re.compile(
    r"""
    ^\s*(?:
        page\s+\d+(?:\s+of\s+\d+)?
        | \d+\s*/\s*\d+
    )\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

DOCUMENT_FIELD_PATTERN = re.compile(
    r"""
    ^\s*(?:
        title
        | document(?:\s+title)?
        | file(?:\s*name)?
        | source(?:\s+file)?
        | recording(?:\s+title)?
        | session(?:\s+title)?
        | lesson\s+title
    )\s*[:\-]\s*.+$
    """,
    re.IGNORECASE | re.VERBOSE,
)

TRANSCRIPT_HEADING_PATTERN = re.compile(
    r"""
    ^\s*(?:
        transcript
        | lesson\s+transcript
        | class\s+transcript
        | meeting\s+transcript
        | raw\s+transcript
        | cleaned\s+transcript
    )
    (?:\s+(?:test|sample|recording|number|no\.?|\#)?\s*\d*)?
    (?:\s*[-:–—].*)?
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

SYNTHETIC_TEST_METADATA_PATTERN = re.compile(
    r"""
    ^\s*.*\b(?:
        synthetic\s+(?:raw\s+)?(?:lesson\s+)?transcript
        | test\s+transcript
        | sample\s+transcript
    )\b.*\b(?:
        test(?:ing)?
        | module\s*\d+
        | edtech
        | benchmark
        | regression
    )\b.*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

DECORATIVE_METADATA_PATTERN = re.compile(
    r"^\s*[-=_*]{3,}\s*$"
)


def _normalise_metadata_text(text: str) -> str:
    """Normalize a heading or filename for conservative comparison."""

    text = re.sub(
        r"\.(?:docx?|pdf|txt|rtf)$",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text.lower(),
    )
    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def _source_name_matches_line(
    line: str,
    source_name: str | None,
) -> bool:
    """
    Return True only when a short line strongly matches the source filename.

    This prevents a filename such as
    ``Transcript_Test_3_Algorithms_Programming_Stress.docx`` from becoming
    lesson evidence while avoiding broad topic-word deletion.
    """

    if not source_name:
        return False

    line_normalized = _normalise_metadata_text(line)
    source_normalized = _normalise_metadata_text(source_name)

    if not line_normalized or not source_normalized:
        return False

    if len(line_normalized) > 180:
        return False

    if line_normalized == source_normalized:
        return True

    line_tokens = set(line_normalized.split())
    source_tokens = set(source_normalized.split())

    if not line_tokens or not source_tokens:
        return False

    overlap = len(line_tokens & source_tokens)
    coverage_of_line = overlap / len(line_tokens)
    coverage_of_source = overlap / len(source_tokens)

    return (
        overlap >= 3
        and coverage_of_line >= 0.85
        and coverage_of_source >= 0.75
    )


def _is_strong_metadata_line(
    line: str,
    source_name: str | None,
) -> bool:
    """Classify only high-confidence document metadata."""

    stripped = line.strip()

    if not stripped:
        return False

    return any(
        (
            _source_name_matches_line(
                stripped,
                source_name,
            ),
            PAGE_METADATA_PATTERN.fullmatch(stripped) is not None,
            DOCUMENT_FIELD_PATTERN.fullmatch(stripped) is not None,
            TRANSCRIPT_HEADING_PATTERN.fullmatch(stripped) is not None,
            SYNTHETIC_TEST_METADATA_PATTERN.fullmatch(stripped) is not None,
            DECORATIVE_METADATA_PATTERN.fullmatch(stripped) is not None,
        )
    )


def remove_document_metadata(
    text: str,
    source_name: str | None = None,
    leading_non_empty_limit: int = 12,
) -> tuple[str, list[str]]:
    """
    Remove high-confidence document metadata before transcript cleaning.

    Rules:
        * filename/title-like lines are removed near the document start;
        * page markers and exact source-name headers are removed anywhere;
        * repeated strong document headers are removed anywhere;
        * normal lesson sentences are preserved.

    The removed lines are returned for the preprocessing audit.
    """

    lines = text.splitlines()
    normalized_counts: dict[str, int] = {}

    for line in lines:
        normalized = _normalise_metadata_text(line)
        if normalized:
            normalized_counts[normalized] = (
                normalized_counts.get(normalized, 0) + 1
            )

    kept_lines: list[str] = []
    removed_lines: list[str] = []
    non_empty_seen = 0

    for line in lines:
        stripped = line.strip()

        if stripped:
            non_empty_seen += 1

        normalized = _normalise_metadata_text(stripped)
        is_leading = (
            non_empty_seen <= leading_non_empty_limit
        )

        source_match = _source_name_matches_line(
            stripped,
            source_name,
        )
        page_marker = (
            PAGE_METADATA_PATTERN.fullmatch(stripped)
            is not None
        )
        repeated_strong_header = (
            bool(normalized)
            and normalized_counts.get(normalized, 0) >= 2
            and _is_strong_metadata_line(
                stripped,
                source_name,
            )
        )

        should_remove = (
            (is_leading and _is_strong_metadata_line(
                stripped,
                source_name,
            ))
            or source_match
            or page_marker
            or repeated_strong_header
        )

        if should_remove:
            if stripped:
                removed_lines.append(stripped)
            continue

        kept_lines.append(line)

    # Remove blank lines left only because a leading metadata block vanished.
    while kept_lines and not kept_lines[0].strip():
        kept_lines.pop(0)

    return (
        "\n".join(kept_lines),
        removed_lines,
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
    Remove consecutive duplicate sentences across the whole transcript.

    This works both:
    - within the same line/paragraph
    - across adjacent lines/paragraphs

    Example:

        Okay.
        Okay.

    becomes:

        Okay.

    Paragraph structure is preserved as much as possible; only the
    duplicate sentence itself is removed.
    """

    removed_count = 0
    cleaned_lines: list[str] = []

    # Keep this outside the line loop so duplicate sentences split
    # across separate DOCX paragraphs can still be detected.
    previous_normalized: str | None = None

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
    source_name: str | None = None,
) -> PreprocessingResult:
    """
    Lightweight deterministic transcript preprocessing.

    This module deliberately avoids semantic rewriting.

    Steps:
        1. Validate input
        2. Remove high-confidence document metadata
        3. Remove combined speaker + timestamp prefixes
        4. Remove standalone timestamps
        5. Remove standalone speaker labels
        6. Remove obvious non-verbal artefacts
        7. Remove uncertainty markers
        8. Remove safe fillers
        9. Compress repeated words
        10. Compress repeated sentences
        11. Normalize whitespace
        12. Return cleaned transcript for semantic chunking
    """

    if not raw_text or not raw_text.strip():

        raise ValueError(
            "Transcript cannot be empty."
        )

    original_characters = len(
        raw_text
    )

    # Metadata must be removed before speaker/timestamp processing so that
    # document titles can never become lesson evidence downstream.
    text, removed_metadata_lines = remove_document_metadata(
        raw_text,
        source_name=source_name,
    )

    metadata_characters_removed = sum(
        len(line)
        for line in removed_metadata_lines
    )

    timestamps_removed = 0
    speaker_labels_removed = 0

    processed_lines: list[str] = []

    # =====================================================
    # LINE-LEVEL CLEANING
    # =====================================================

    for line in text.splitlines():

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
        timestamps_removed=timestamps_removed,
        speaker_labels_removed=speaker_labels_removed,
        fillers_removed=fillers_removed,
        artefacts_removed=artefacts_removed,
        uncertainty_markers_removed=uncertainty_markers_removed,
        repeated_words_removed=repeated_words_removed,
        repeated_sentences_removed=repeated_sentences_removed,
        metadata_lines_removed=len(removed_metadata_lines),
        metadata_characters_removed=metadata_characters_removed,
    )

    return PreprocessingResult(
        cleaned_text=text,
        warnings=warnings,
        removed_metadata_lines=removed_metadata_lines,
        stats=stats,
    )