from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ExtractionMethod = Literal[
    "keyword",
    "embedding",
    "keyword_embedding",
]

ChunkClassification = Literal[
    "official_aqa_topic",
    "mixed_official_and_unmapped",
    "cs_related_unmapped",
    "continuation_no_new_topic",
    "no_topic",
]

UnmappedDetectionMethod = Literal[
    "lexical",
    "semantic",
    "lexical_semantic",
]

TopicRole = Literal[
    "primary",
    "supporting",
]


class RawTopicCandidate(BaseModel):
    """
    Official AQA topic candidate before the final relevance decision.
    """

    concept_id: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    domain: str = Field(min_length=1)

    official_reference: str = Field(min_length=1)
    chapter_reference: str = Field(min_length=1)
    official_title: str = Field(min_length=1)
    paper: str = Field(min_length=1)
    source_pages: list[int] = Field(default_factory=list)

    confidence: float = Field(ge=0.0, le=1.0)
    keyword_score: float = Field(ge=0.0, le=1.0)
    semantic_score: float = Field(ge=-1.0, le=1.0)
    salience_score: float = Field(ge=0.0, le=1.0)

    extraction_method: ExtractionMethod

    matched_aliases: list[str] = Field(default_factory=list)
    total_alias_hits: int = Field(default=0, ge=0)
    evidence_sentence_count: int = Field(default=0, ge=0)
    single_word_alias_only: bool = False

    evidence: list[str] = Field(default_factory=list)
    parent_concept_id: str | None = None


class TopicCandidate(RawTopicCandidate):
    """
    Candidate after the official-topic relevance filter.
    """

    cs_relevance_score: float = Field(ge=0.0, le=1.0)
    cs_relevant: bool


class UnmappedCSSignal(BaseModel):
    """
    Evidence that a chunk contains a Computer Science concept which was not
    confidently covered by an official AQA catalogue candidate.

    rough_topic is intentionally not an AQA mapping. It is a generic label
    that can later be reviewed or refined by the fallback stage.
    """

    rough_topic: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    score: float = Field(ge=-1.0, le=1.0)
    evidence: str = Field(min_length=1)
    matched_aliases: list[str] = Field(default_factory=list)
    detection_method: UnmappedDetectionMethod


class ChunkTopicResult(BaseModel):
    """
    Complete Module 3 output for one transcript chunk.
    """

    chunk_id: int = Field(ge=1)
    source_word_count: int = Field(default=0, ge=0)

    classification: ChunkClassification = "no_topic"

    # A continuation-only chunk does not create a new topic.
    is_cs_relevant: bool
    creates_new_topic: bool = False

    cs_relevance_score: float = Field(ge=0.0, le=1.0)

    topic_candidates: list[TopicCandidate] = Field(default_factory=list)
    rejected_candidates: list[TopicCandidate] = Field(default_factory=list)

    has_unmapped_cs_content: bool = False
    unmapped_cs_signals: list[UnmappedCSSignal] = Field(default_factory=list)

    continuation_of_chunk_id: int | None = Field(default=None, ge=1)

    requires_llm_fallback: bool = False
    notes: list[str] = Field(default_factory=list)


class MergedTopic(BaseModel):
    """
    One official AQA topic merged across one or more transcript chunks.

    confidence remains the extraction confidence. ranking_score is used only
    to order lesson topics more realistically by semantic strength, salience
    and coverage rather than by repeated keyword matches alone.
    """

    concept_id: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    domain: str = Field(min_length=1)

    official_reference: str = Field(min_length=1)
    chapter_reference: str = Field(min_length=1)
    official_title: str = Field(min_length=1)
    paper: str = Field(min_length=1)
    source_pages: list[int] = Field(default_factory=list)

    confidence: float = Field(ge=0.0, le=1.0)
    ranking_score: float = Field(ge=0.0, le=1.0)
    topic_role: TopicRole

    source_chunk_ids: list[int] = Field(default_factory=list)
    support_span_count: int = Field(default=1, ge=1)

    mean_semantic_score: float = Field(ge=-1.0, le=1.0)
    mean_keyword_score: float = Field(ge=0.0, le=1.0)
    mean_salience_score: float = Field(ge=0.0, le=1.0)
    coverage_score: float = Field(ge=0.0, le=1.0)

    evidence: list[str] = Field(default_factory=list)
    supporting_candidate_count: int = Field(ge=1)


class Module3Result(BaseModel):
    """
    Complete output of Module 3.
    """

    chunk_results: list[ChunkTopicResult]
    merged_topics: list[MergedTopic]

    total_chunks: int = Field(ge=0)
    cs_relevant_chunks: int = Field(ge=0)
    non_cs_chunks: int = Field(ge=0)

    official_topic_chunks: int = Field(default=0, ge=0)
    mixed_official_unmapped_chunks: int = Field(default=0, ge=0)
    unmapped_cs_chunks: int = Field(default=0, ge=0)
    continuation_chunks: int = Field(default=0, ge=0)
    no_topic_chunks: int = Field(default=0, ge=0)

    llm_fallback_chunk_ids: list[int] = Field(default_factory=list)

    embedding_model: str
    candidate_keep_threshold: float = Field(ge=0.0, le=1.0)