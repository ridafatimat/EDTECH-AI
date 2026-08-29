from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from app.services.topic_mapping_memory_service import (
    TopicMappingMemoryConfig,
    TopicMappingMemoryService,
)


class FakeMemoryRepository:
    def __init__(self, candidates):
        self.candidates = candidates
        self.marked_used = []
        self.last_lookup = None

    def find_reusable_candidates(self, *, normalized_topic, spec_version, limit):
        self.last_lookup = (normalized_topic, spec_version, limit)
        return list(self.candidates[:limit])

    def mark_used(self, record_id):
        self.marked_used.append(record_id)
        for item in self.candidates:
            if item.id == record_id:
                item.hit_count += 1
                return item
        raise KeyError(record_id)


class FakeDecisionLogRepository:
    def __init__(self):
        self.rows = []

    def log(self, **kwargs):
        self.rows.append(kwargs)
        return kwargs


def make_memory(
    record_id,
    *,
    evidence_text,
    reviewer_reason,
    mapped_concept_id,
    decision="mapped",
    confidence=0.95,
    validation_status="human_corrected",
):
    return SimpleNamespace(
        id=record_id,
        evidence_text=evidence_text,
        reviewer_reason=reviewer_reason,
        mapped_concept_id=mapped_concept_id,
        decision=decision,
        confidence=confidence,
        validation_status=validation_status,
        hit_count=0,
    )


def make_embedder(vectors):
    def fake_embedder(texts, model_name, batch_size):
        del model_name, batch_size
        rows = []
        for text in texts:
            vector = np.asarray(vectors[text], dtype=np.float32)
            vector = vector / np.linalg.norm(vector)
            rows.append(vector)
        return np.vstack(rows)

    return fake_embedder


def config():
    return TopicMappingMemoryConfig(
        evidence_similarity_threshold=0.82,
        reviewer_reason_similarity_threshold=0.60,
        combined_similarity_threshold=0.80,
        human_corrected_evidence_threshold=0.60,
        human_corrected_reason_threshold=0.60,
        human_corrected_combined_threshold=0.60,
        human_corrected_exact_evidence_threshold=0.95,
        reviewer_reason_weight=0.25,
        minimum_score_margin=0.04,
        minimum_evidence_characters=10,
    )


def test_strong_memory_hit_and_reuse_logging():
    new_evidence = (
        "The algorithm checks each item in order until the target is found."
    )
    old_evidence = (
        "Linear search examines values sequentially until the target is found."
    )
    correction_reason = (
        "The transcript describes sequential item checking, not halving."
    )

    memory = make_memory(
        11,
        evidence_text=old_evidence,
        reviewer_reason=correction_reason,
        mapped_concept_id="linear_search",
    )
    repo = FakeMemoryRepository([memory])
    log_repo = FakeDecisionLogRepository()

    vectors = {
        new_evidence: [1.0, 0.0, 0.0],
        old_evidence: [0.99, 0.05, 0.0],
        correction_reason: [0.92, 0.12, 0.0],
    }

    service = TopicMappingMemoryService(
        repo,
        log_repo,
        config=config(),
        embedding_function=make_embedder(vectors),
    )

    result = service.evaluate_and_record(
        normalized_topic="searching algorithms",
        new_evidence=new_evidence,
        spec_version="AQA-8525-v1.2-2022-11-29",
        pipeline_run_id="test-run-1",
        source_transcript="test.docx",
        source_chunk_ids=[1],
    )

    assert result.is_hit
    assert result.source_memory_id == 11
    assert result.matched_memory.mapped_concept_id == "linear_search"
    assert repo.marked_used == [11]
    assert memory.hit_count == 1
    assert len(log_repo.rows) == 1
    assert log_repo.rows[0]["action"] == "reuse"
    assert log_repo.rows[0]["source_memory_id"] == 11
    assert log_repo.rows[0]["spec_version"] == "AQA-8525-v1.2-2022-11-29"


def test_mismatch_falls_back_to_qdrant():
    new_evidence = (
        "The middle value is checked and half the search space is discarded."
    )
    old_evidence = (
        "Linear search examines values sequentially until the target is found."
    )
    correction_reason = (
        "The transcript describes sequential item checking, not halving."
    )

    memory = make_memory(
        12,
        evidence_text=old_evidence,
        reviewer_reason=correction_reason,
        mapped_concept_id="linear_search",
    )
    repo = FakeMemoryRepository([memory])
    log_repo = FakeDecisionLogRepository()

    vectors = {
        new_evidence: [0.0, 1.0, 0.0],
        old_evidence: [1.0, 0.0, 0.0],
        correction_reason: [0.9, 0.1, 0.0],
    }

    service = TopicMappingMemoryService(
        repo,
        log_repo,
        config=config(),
        embedding_function=make_embedder(vectors),
    )

    result = service.evaluate_and_record(
        normalized_topic="searching algorithms",
        new_evidence=new_evidence,
        spec_version="AQA-8525-v1.2-2022-11-29",
    )

    assert not result.is_hit
    assert result.status == "miss"
    assert repo.marked_used == []
    assert log_repo.rows[0]["action"] == "memory_miss"


def test_conflicting_memories_are_ambiguous():
    new_evidence = "The search method checks values in a structured way."
    evidence_a = "Search method A uses a structured sequence of checks."
    evidence_b = "Search method B also uses a structured sequence of checks."

    memory_a = make_memory(
        21,
        evidence_text=evidence_a,
        reviewer_reason=None,
        mapped_concept_id="linear_search",
        validation_status="validated",
    )
    memory_b = make_memory(
        22,
        evidence_text=evidence_b,
        reviewer_reason=None,
        mapped_concept_id="binary_search",
        validation_status="validated",
    )

    repo = FakeMemoryRepository([memory_a, memory_b])

    vectors = {
        new_evidence: [1.0, 0.0, 0.0],
        evidence_a: [0.99, 0.08, 0.0],
        evidence_b: [0.988, 0.10, 0.0],
    }

    service = TopicMappingMemoryService(
        repo,
        config=config(),
        embedding_function=make_embedder(vectors),
    )

    result = service.evaluate(
        normalized_topic="searching algorithms",
        new_evidence=new_evidence,
        spec_version="AQA-8525-v1.2-2022-11-29",
    )

    assert result.status == "ambiguous"
    assert not result.is_hit



def test_exact_evidence_human_correction_does_not_fail_on_short_reason():
    """Regression: exact evidence may only moderately match a short reason."""
    new_evidence = "Full transcript evidence about subroutines, functions, parameters, return values and local variables."
    old_evidence = "Full transcript evidence about subroutines, functions, parameters, return values and local variables."
    correction_reason = "Functions and procedures belong under the subroutines specification section."

    memory = make_memory(
        31,
        evidence_text=old_evidence,
        reviewer_reason=correction_reason,
        mapped_concept_id="aqa_3_2_10_subroutines",
    )
    repo = FakeMemoryRepository([memory])

    # evidence similarity = 1.0; reason similarity ~0.558; combined > 0.80.
    vectors = {
        new_evidence: [1.0, 0.0, 0.0],
        old_evidence: [1.0, 0.0, 0.0],
        correction_reason: [0.558, 0.82984, 0.0],
    }

    service = TopicMappingMemoryService(
        repo,
        config=config(),
        embedding_function=make_embedder(vectors),
    )

    result = service.evaluate(
        normalized_topic="subroutine statements",
        new_evidence=new_evidence,
        spec_version="AQA-8525-v1.2-2022-11-29",
    )

    assert result.is_hit
    assert result.source_memory_id == 31
    assert result.reviewer_reason_similarity is not None
    assert result.reviewer_reason_similarity < 0.60
    assert result.combined_similarity >= 0.80


def test_reason_still_matters_through_combined_score():
    """Moderate evidence + poor reason alignment must still fall back."""
    new_evidence = "A future transcript with only moderately similar evidence."
    old_evidence = "Stored evidence from the corrected historical case."
    correction_reason = "A correction reason describing a different distinction."

    memory = make_memory(
        32,
        evidence_text=old_evidence,
        reviewer_reason=correction_reason,
        mapped_concept_id="corrected_topic",
    )
    repo = FakeMemoryRepository([memory])

    vectors = {
        new_evidence: [1.0, 0.0, 0.0],
        # ~0.85 evidence similarity: passes evidence gate.
        old_evidence: [0.85, 0.5267827, 0.0],
        # ~0.20 reason similarity: combined score falls below 0.80.
        correction_reason: [0.20, 0.9797959, 0.0],
    }

    service = TopicMappingMemoryService(
        repo,
        config=config(),
        embedding_function=make_embedder(vectors),
    )

    result = service.evaluate(
        normalized_topic="some topic",
        new_evidence=new_evidence,
        spec_version="AQA-8525-v1.2-2022-11-29",
    )

    assert result.status == "miss"
    assert not result.is_hit
    assert result.evidence_similarity is not None
    assert result.evidence_similarity >= 0.82
    assert result.combined_similarity is not None
    assert result.combined_similarity < 0.80


def test_human_corrected_paraphrase_reuse_without_lowering_normal_gates():
    """Regression for the real subroutine paraphrase diagnostic (~0.61 scores)."""
    new_evidence = "A paraphrased future lesson about functions, procedures, parameters, returns and local variables."
    old_evidence = "A longer historical lesson chunk that taught subroutines alongside other programming ideas."
    correction_reason = "Functions, procedures, parameters, return values and local variables belong under AQA 3.2.10."

    memory = make_memory(
        41,
        evidence_text=old_evidence,
        reviewer_reason=correction_reason,
        mapped_concept_id="aqa_3_2_10_subroutines",
        validation_status="human_corrected",
    )
    repo = FakeMemoryRepository([memory])

    # Mirrors the production diagnostic: evidence ~0.6092, reason ~0.6010,
    # combined ~0.6071. Strict 0.82/0.80 gates fail, but the scoped
    # human-corrected paraphrase path should pass.
    ev = 0.6092
    rs = 0.6010
    vectors = {
        new_evidence: [1.0, 0.0, 0.0],
        old_evidence: [ev, float((1.0 - ev * ev) ** 0.5), 0.0],
        correction_reason: [rs, float((1.0 - rs * rs) ** 0.5), 0.0],
    }

    service = TopicMappingMemoryService(
        repo,
        config=config(),
        embedding_function=make_embedder(vectors),
    )

    result = service.evaluate(
        normalized_topic="subroutine statements",
        new_evidence=new_evidence,
        spec_version="AQA-8525-v1.2-2022-11-29",
    )

    assert result.is_hit
    assert result.source_memory_id == 41
    assert result.evidence_similarity is not None
    assert 0.60 <= result.evidence_similarity < 0.82
    assert result.reviewer_reason_similarity is not None
    assert result.reviewer_reason_similarity >= 0.60
    assert result.combined_similarity is not None
    assert 0.60 <= result.combined_similarity < 0.80


def test_paraphrase_path_is_not_available_to_normal_approved_memory():
    new_evidence = "A moderately similar future lesson."
    old_evidence = "A moderately similar historical lesson."
    reason = "A semantically aligned note."

    memory = make_memory(
        42,
        evidence_text=old_evidence,
        reviewer_reason=reason,
        mapped_concept_id="some_topic",
        validation_status="validated",
    )
    repo = FakeMemoryRepository([memory])

    ev = 0.61
    rs = 0.61
    vectors = {
        new_evidence: [1.0, 0.0, 0.0],
        old_evidence: [ev, float((1.0 - ev * ev) ** 0.5), 0.0],
        reason: [rs, float((1.0 - rs * rs) ** 0.5), 0.0],
    }

    service = TopicMappingMemoryService(
        repo,
        config=config(),
        embedding_function=make_embedder(vectors),
    )

    result = service.evaluate(
        normalized_topic="some topic",
        new_evidence=new_evidence,
        spec_version="AQA-8525-v1.2-2022-11-29",
    )

    assert result.status == "miss"
    assert not result.is_hit


def test_human_corrected_paraphrase_requires_reason_alignment():
    new_evidence = "A moderately similar future lesson."
    old_evidence = "A moderately similar historical lesson."
    correction_reason = "A correction reason that does not align well enough."

    memory = make_memory(
        43,
        evidence_text=old_evidence,
        reviewer_reason=correction_reason,
        mapped_concept_id="corrected_topic",
        validation_status="human_corrected",
    )
    repo = FakeMemoryRepository([memory])

    ev = 0.65
    rs = 0.40
    vectors = {
        new_evidence: [1.0, 0.0, 0.0],
        old_evidence: [ev, float((1.0 - ev * ev) ** 0.5), 0.0],
        correction_reason: [rs, float((1.0 - rs * rs) ** 0.5), 0.0],
    }

    service = TopicMappingMemoryService(
        repo,
        config=config(),
        embedding_function=make_embedder(vectors),
    )

    result = service.evaluate(
        normalized_topic="some topic",
        new_evidence=new_evidence,
        spec_version="AQA-8525-v1.2-2022-11-29",
    )

    assert result.status == "miss"
    assert not result.is_hit



def test_human_corrected_near_identical_evidence_can_override_negative_reason_similarity():
    """Regression for beyond-GCSE corrections: exact evidence must reuse safely."""
    new_evidence = "Identical transcript evidence about Big O and A-level complexity."
    old_evidence = "Identical transcript evidence about Big O and A-level complexity."
    correction_reason = "This content is beyond the AQA GCSE specification."

    memory = make_memory(
        51,
        evidence_text=old_evidence,
        reviewer_reason=correction_reason,
        mapped_concept_id=None,
        decision="out_of_syllabus",
        validation_status="human_corrected",
    )
    repo = FakeMemoryRepository([memory])

    # Mirrors production diagnostic: evidence=1.0, reason≈-0.0452,
    # combined≈0.7387. Strict combined and paraphrase gates both fail, but
    # the near-identical human-corrected evidence path should pass.
    rs = -0.0452
    vectors = {
        new_evidence: [1.0, 0.0, 0.0],
        old_evidence: [1.0, 0.0, 0.0],
        correction_reason: [rs, float((1.0 - rs * rs) ** 0.5), 0.0],
    }

    service = TopicMappingMemoryService(
        repo,
        config=config(),
        embedding_function=make_embedder(vectors),
    )

    result = service.evaluate(
        normalized_topic="big o notation",
        new_evidence=new_evidence,
        spec_version="AQA-8525-v1.2-2022-11-29",
    )

    assert result.is_hit
    assert result.source_memory_id == 51
    assert result.evidence_similarity is not None
    assert result.evidence_similarity >= 0.95
    assert result.reviewer_reason_similarity is not None
    assert result.reviewer_reason_similarity < 0.0
    assert result.combined_similarity is not None
    assert result.combined_similarity < 0.80


def test_human_corrected_non_identical_evidence_still_needs_paraphrase_gates():
    """A 0.94 evidence match cannot bypass a poorly aligned correction reason."""
    new_evidence = "A similar but not near-identical future transcript."
    old_evidence = "Historical evidence with related but different wording."
    correction_reason = "A correction reason that points in another direction."

    memory = make_memory(
        52,
        evidence_text=old_evidence,
        reviewer_reason=correction_reason,
        mapped_concept_id=None,
        decision="out_of_syllabus",
        validation_status="human_corrected",
    )
    repo = FakeMemoryRepository([memory])

    ev = 0.94
    rs = 0.10
    vectors = {
        new_evidence: [1.0, 0.0, 0.0],
        old_evidence: [ev, float((1.0 - ev * ev) ** 0.5), 0.0],
        correction_reason: [rs, float((1.0 - rs * rs) ** 0.5), 0.0],
    }

    service = TopicMappingMemoryService(
        repo,
        config=config(),
        embedding_function=make_embedder(vectors),
    )

    result = service.evaluate(
        normalized_topic="big o notation",
        new_evidence=new_evidence,
        spec_version="AQA-8525-v1.2-2022-11-29",
    )

    assert result.status == "miss"
    assert not result.is_hit


def run_all_tests():
    test_strong_memory_hit_and_reuse_logging()
    test_mismatch_falls_back_to_qdrant()
    test_conflicting_memories_are_ambiguous()
    test_exact_evidence_human_correction_does_not_fail_on_short_reason()
    test_reason_still_matters_through_combined_score()
    test_human_corrected_paraphrase_reuse_without_lowering_normal_gates()
    test_paraphrase_path_is_not_available_to_normal_approved_memory()
    test_human_corrected_paraphrase_requires_reason_alignment()
    test_human_corrected_near_identical_evidence_can_override_negative_reason_similarity()
    test_human_corrected_non_identical_evidence_still_needs_paraphrase_gates()
    print("Topic mapping memory compatibility tests passed")


if __name__ == "__main__":
    run_all_tests()
