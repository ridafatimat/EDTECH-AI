from __future__ import annotations

from app.services.detected_topic_edit_reason_validator import (
    ReviewerReasonContextValidator,
    ReviewerReasonValidationRequest,
)


class FakeProvider:
    """
    Deterministic provider for isolated policy tests.

    The production validator contains no topic-specific logic.
    """

    def __init__(self, response):
        self.response = dict(response)
        self.calls = []

    def validate_edit_context(self, **kwargs):
        self.calls.append(dict(kwargs))
        return dict(self.response)


def make_request(
    *,
    edit_action="remove_topic",
    source_topic="Sorting algorithms",
    source_role="supporting",
    target_topic=None,
    target_role=None,
):
    return ReviewerReasonValidationRequest(
        edit_action=edit_action,
        source_topic=source_topic,
        source_role=source_role,
        target_topic=target_topic,
        target_role=target_role,
        reviewer_reason=(
            "The topic is only mentioned as a prerequisite and is not "
            "independently taught in this lesson."
        ),
        stored_evidence=(
            "Binary search needs ordered data, so an unsorted list would "
            "need sorting before the search begins."
        ),
        current_evidence=(
            "Binary search requires a sorted collection; unordered input "
            "would first need to be sorted."
        ),
    )


def expect_value_error(callback):
    try:
        callback()
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


def main():
    # --------------------------------------------------------------
    # 1. Clear compatible removal context -> safe
    # --------------------------------------------------------------
    provider = FakeProvider(
        {
            "decision": "compatible",
            "rationale_still_applies": True,
            "same_teaching_context": True,
            "independent_teaching_detected": False,
            "confidence": 0.97,
            "explanation": (
                "The topic remains incidental and is not independently taught."
            ),
        }
    )

    validator = ReviewerReasonContextValidator(
        provider=provider
    )

    result = validator.validate(make_request())

    assert result.decision == "compatible"
    assert result.safe_for_automatic_reuse is True

    # --------------------------------------------------------------
    # 2. Actual independent teaching -> removal must NOT reuse
    # --------------------------------------------------------------
    provider = FakeProvider(
        {
            "decision": "incompatible",
            "rationale_still_applies": False,
            "same_teaching_context": False,
            "independent_teaching_detected": True,
            "confidence": 0.99,
            "explanation": (
                "The topic is independently explained in the current lesson."
            ),
        }
    )

    validator = ReviewerReasonContextValidator(
        provider=provider
    )

    result = validator.validate(make_request())

    assert result.safe_for_automatic_reuse is False

    # --------------------------------------------------------------
    # 3. Uncertain always abstains
    # --------------------------------------------------------------
    provider = FakeProvider(
        {
            "decision": "uncertain",
            "rationale_still_applies": True,
            "same_teaching_context": True,
            "independent_teaching_detected": False,
            "confidence": 0.93,
            "explanation": (
                "Evidence is insufficient to decide whether teaching is "
                "independent or incidental."
            ),
        }
    )

    result = ReviewerReasonContextValidator(
        provider=provider
    ).validate(make_request())

    assert result.safe_for_automatic_reuse is False

    # --------------------------------------------------------------
    # 4. Low provider confidence -> abstain even when compatible
    # --------------------------------------------------------------
    provider = FakeProvider(
        {
            "decision": "compatible",
            "rationale_still_applies": True,
            "same_teaching_context": True,
            "independent_teaching_detected": False,
            "confidence": 0.72,
            "explanation": "Likely compatible, but confidence is limited.",
        }
    )

    result = ReviewerReasonContextValidator(
        provider=provider
    ).validate(make_request())

    assert result.safe_for_automatic_reuse is False

    # --------------------------------------------------------------
    # 5. Contradictory provider payload -> abstain
    # --------------------------------------------------------------
    provider = FakeProvider(
        {
            "decision": "compatible",
            "rationale_still_applies": False,
            "same_teaching_context": True,
            "independent_teaching_detected": False,
            "confidence": 0.98,
            "explanation": (
                "Provider says compatible but also says the rationale no "
                "longer applies."
            ),
        }
    )

    result = ReviewerReasonContextValidator(
        provider=provider
    ).validate(make_request())

    assert result.safe_for_automatic_reuse is False

    # --------------------------------------------------------------
    # 6. Removal marked compatible but independent teaching detected
    #    -> must abstain
    # --------------------------------------------------------------
    provider = FakeProvider(
        {
            "decision": "compatible",
            "rationale_still_applies": True,
            "same_teaching_context": True,
            "independent_teaching_detected": True,
            "confidence": 0.99,
            "explanation": (
                "The context is similar, but the topic is independently taught."
            ),
        }
    )

    result = ReviewerReasonContextValidator(
        provider=provider
    ).validate(make_request())

    assert result.safe_for_automatic_reuse is False

    # --------------------------------------------------------------
    # 7. change_role can reuse when the same rationale/focus applies
    # --------------------------------------------------------------
    provider = FakeProvider(
        {
            "decision": "compatible",
            "rationale_still_applies": True,
            "same_teaching_context": True,
            "independent_teaching_detected": None,
            "confidence": 0.96,
            "explanation": (
                "The current lesson has the same main teaching emphasis."
            ),
        }
    )

    role_request = ReviewerReasonValidationRequest(
        edit_action="change_role",
        source_topic="Time efficiency of algorithms",
        source_role="supporting",
        target_topic="Time efficiency of algorithms",
        target_role="primary",
        reviewer_reason=(
            "Time efficiency is the central teaching objective rather than "
            "a supporting comparison."
        ),
        stored_evidence=(
            "The lesson repeatedly compares execution efficiency and explains "
            "why one algorithm needs fewer operations."
        ),
        current_evidence=(
            "Most of the new lesson evaluates algorithm efficiency and "
            "justifies which solution is faster."
        ),
    )

    result = ReviewerReasonContextValidator(
        provider=provider
    ).validate(role_request)

    assert result.safe_for_automatic_reuse is True

    # --------------------------------------------------------------
    # 8. add_topic can reuse only when the missed teaching rationale
    #    still applies strongly
    # --------------------------------------------------------------
    provider = FakeProvider(
        {
            "decision": "compatible",
            "rationale_still_applies": True,
            "same_teaching_context": True,
            "independent_teaching_detected": True,
            "confidence": 0.97,
            "explanation": (
                "The current evidence again independently teaches the topic."
            ),
        }
    )

    add_request = ReviewerReasonValidationRequest(
        edit_action="add_topic",
        source_topic=None,
        source_role=None,
        target_topic="Boolean operations",
        target_role="supporting",
        reviewer_reason=(
            "Boolean operators are explicitly taught but were missed from "
            "the detected topic list."
        ),
        stored_evidence=(
            "AND OR and NOT are explained with truth values and examples."
        ),
        current_evidence=(
            "Students again learn AND OR and NOT using Boolean expressions."
        ),
    )

    result = ReviewerReasonContextValidator(
        provider=provider
    ).validate(add_request)

    assert result.safe_for_automatic_reuse is True

    # --------------------------------------------------------------
    # 9. Invalid action rejected
    # --------------------------------------------------------------
    expect_value_error(
        lambda: ReviewerReasonContextValidator(
            provider=provider
        ).validate(
            ReviewerReasonValidationRequest(
                edit_action="delete_everything",
                source_topic="X",
                source_role="supporting",
                target_topic=None,
                target_role=None,
                reviewer_reason="Reason",
                stored_evidence="Stored evidence",
                current_evidence="Current evidence",
            )
        )
    )

    # --------------------------------------------------------------
    # 10. Missing human reason rejected
    # --------------------------------------------------------------
    expect_value_error(
        lambda: ReviewerReasonContextValidator(
            provider=provider
        ).validate(
            ReviewerReasonValidationRequest(
                edit_action="remove_topic",
                source_topic="X",
                source_role="supporting",
                target_topic=None,
                target_role=None,
                reviewer_reason="",
                stored_evidence="Stored evidence",
                current_evidence="Current evidence",
            )
        )
    )

    print("Reviewer-reason context validator isolated tests passed")


if __name__ == "__main__":
    main()
