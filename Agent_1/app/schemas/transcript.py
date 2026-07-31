from pydantic import BaseModel, Field


class PreprocessingStats(BaseModel):
    """
    Statistics from lightweight deterministic preprocessing.
    """

    original_characters: int
    cleaned_characters: int

    timestamps_removed: int = 0
    speaker_labels_removed: int = 0
    fillers_removed: int = 0
    artefacts_removed: int = 0
    uncertainty_markers_removed: int = 0

    repeated_words_removed: int = 0
    repeated_sentences_removed: int = 0

    # High-confidence non-lesson document metadata removed before chunking.
    metadata_lines_removed: int = 0
    metadata_characters_removed: int = 0


class PreprocessingResult(BaseModel):
    """
    Final output of Agent 1 Module 1.

    The cleaned text is passed directly to semantic chunking.
    """

    cleaned_text: str

    # Non-fatal issues which later stages may want to know about.
    warnings: list[str] = Field(
        default_factory=list
    )

    # Preserved in the audit output so removals can be reviewed. These values
    # must never be passed into semantic chunking or topic extraction.
    removed_metadata_lines: list[str] = Field(
        default_factory=list
    )

    stats: PreprocessingStats