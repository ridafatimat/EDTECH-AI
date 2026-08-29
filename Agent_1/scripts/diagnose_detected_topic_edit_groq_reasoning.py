from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from app.services.detected_topic_edit_groq_reason_provider import (
    GroqReviewerReasonProvider,
)
from app.services.detected_topic_edit_reason_validator import (
    ReviewerReasonContextValidator,
    ReviewerReasonValidationRequest,
)


load_dotenv()


@dataclass(frozen=True)
class DiagnosticCase:
    name: str
    request: ReviewerReasonValidationRequest
    expected_decisions: tuple[str, ...]
    expected_safe_reuse: bool


def case(
    *,
    name: str,
    edit_action: str,
    source_topic: str | None,
    source_role: str | None,
    target_topic: str | None,
    target_role: str | None,
    reviewer_reason: str,
    stored_evidence: str,
    current_evidence: str,
    expected_decisions: tuple[str, ...],
    expected_safe_reuse: bool,
) -> DiagnosticCase:
    return DiagnosticCase(
        name=name,
        request=ReviewerReasonValidationRequest(
            edit_action=edit_action,
            source_topic=source_topic,
            source_role=source_role,
            target_topic=target_topic,
            target_role=target_role,
            reviewer_reason=reviewer_reason,
            stored_evidence=stored_evidence,
            current_evidence=current_evidence,
        ),
        expected_decisions=expected_decisions,
        expected_safe_reuse=expected_safe_reuse,
    )


def main() -> None:
    provider = GroqReviewerReasonProvider()

    validator = ReviewerReasonContextValidator(
        provider=provider
    )

    print("=" * 100)
    print("STEP 4.7 — REAL GROQ REVIEWER-REASON DIAGNOSTICS")
    print("=" * 100)
    print()
    print("Groq model:", provider.model_name)
    print(
        "Validator confidence floor:",
        validator.config.minimum_provider_confidence,
    )
    print()
    print(
        "READ-ONLY DIAGNOSTIC: no PostgreSQL writes, "
        "no Module 3 edits, no Streamlit changes."
    )

    cases = [

        # --------------------------------------------------------------
        # REMOVE TOPIC — sorting
        # --------------------------------------------------------------
        case(
            name="remove-sorting-positive",
            edit_action="remove_topic",
            source_topic="Sorting algorithms",
            source_role="supporting",
            target_topic=None,
            target_role=None,
            reviewer_reason=(
                "Sorting is only mentioned as a prerequisite for binary "
                "search; no sorting algorithm is actually taught."
            ),
            stored_evidence=(
                "Binary search requires ordered data. If the list is "
                "unsorted, some form of sorting would be needed before "
                "the search can begin."
            ),
            current_evidence=(
                "A binary search only works on ordered items, so an "
                "unordered list would need to be sorted first before "
                "binary search is used."
            ),
            expected_decisions=("compatible",),
            expected_safe_reuse=True,
        ),

        case(
            name="remove-sorting-negative-real-teaching",
            edit_action="remove_topic",
            source_topic="Sorting algorithms",
            source_role="supporting",
            target_topic=None,
            target_role=None,
            reviewer_reason=(
                "Sorting is only mentioned as a prerequisite for binary "
                "search; no sorting algorithm is actually taught."
            ),
            stored_evidence=(
                "Binary search requires ordered data. If the list is "
                "unsorted, some form of sorting would be needed before "
                "the search can begin."
            ),
            current_evidence=(
                "Bubble sort compares adjacent values and swaps them when "
                "they are in the wrong order. Repeated passes continue "
                "until no swaps remain, and students trace each pass."
            ),
            expected_decisions=(
                "incompatible",
                "uncertain",
            ),
            expected_safe_reuse=False,
        ),

        # --------------------------------------------------------------
        # REMOVE TOPIC — arithmetic
        # --------------------------------------------------------------
        case(
            name="remove-arithmetic-positive",
            edit_action="remove_topic",
            source_topic="Arithmetic operations",
            source_role="supporting",
            target_topic=None,
            target_role=None,
            reviewer_reason=(
                "Arithmetic is only used to calculate and update binary-"
                "search pointers and the midpoint; arithmetic operations "
                "are not independently taught."
            ),
            stored_evidence=(
                "The midpoint is calculated using left plus right and "
                "integer division by two, then pointers are adjusted."
            ),
            current_evidence=(
                "To perform binary search, add the left and right pointer "
                "positions, use integer division by two for the midpoint, "
                "and move a pointer."
            ),
            expected_decisions=("compatible",),
            expected_safe_reuse=True,
        ),

        case(
            name="remove-arithmetic-negative-real-teaching",
            edit_action="remove_topic",
            source_topic="Arithmetic operations",
            source_role="supporting",
            target_topic=None,
            target_role=None,
            reviewer_reason=(
                "Arithmetic is only used to calculate and update binary-"
                "search pointers and the midpoint; arithmetic operations "
                "are not independently taught."
            ),
            stored_evidence=(
                "The midpoint is calculated using left plus right and "
                "integer division by two, then pointers are adjusted."
            ),
            current_evidence=(
                "This lesson teaches arithmetic operators including "
                "addition, subtraction, multiplication, division, integer "
                "division and modulo, with worked expressions."
            ),
            expected_decisions=(
                "incompatible",
                "uncertain",
            ),
            expected_safe_reuse=False,
        ),

        # --------------------------------------------------------------
        # CHANGE ROLE
        # --------------------------------------------------------------
        case(
            name="change-role-efficiency-positive",
            edit_action="change_role",
            source_topic="Time efficiency of algorithms",
            source_role="supporting",
            target_topic="Time efficiency of algorithms",
            target_role="primary",
            reviewer_reason=(
                "Time efficiency is the central teaching objective rather "
                "than a supporting comparison."
            ),
            stored_evidence=(
                "The lesson repeatedly compares two algorithms solving the "
                "same problem and explains why one needs fewer operations "
                "and is faster."
            ),
            current_evidence=(
                "Most of the lesson compares algorithms for the same task "
                "and explains their relative execution time and number of "
                "operations."
            ),
            expected_decisions=("compatible",),
            expected_safe_reuse=True,
        ),

        case(
            name="change-role-efficiency-negative-incidental",
            edit_action="change_role",
            source_topic="Time efficiency of algorithms",
            source_role="supporting",
            target_topic="Time efficiency of algorithms",
            target_role="primary",
            reviewer_reason=(
                "Time efficiency is the central teaching objective rather "
                "than a supporting comparison."
            ),
            stored_evidence=(
                "The lesson repeatedly compares two algorithms solving the "
                "same problem and explains why one needs fewer operations "
                "and is faster."
            ),
            current_evidence=(
                "The teacher briefly notes that binary search is faster "
                "than linear search before returning to midpoint "
                "calculation and pointer updates."
            ),
            expected_decisions=(
                "incompatible",
                "uncertain",
            ),
            expected_safe_reuse=False,
        ),

        # --------------------------------------------------------------
        # ADD TOPIC
        # --------------------------------------------------------------
        case(
            name="add-boolean-positive",
            edit_action="add_topic",
            source_topic=None,
            source_role=None,
            target_topic="Boolean operations",
            target_role="supporting",
            reviewer_reason=(
                "Boolean operators are explicitly taught but were missed "
                "from the detected topic list."
            ),
            stored_evidence=(
                "The lesson explicitly teaches Boolean AND OR and NOT "
                "operators, including truth conditions and worked examples."
            ),
            current_evidence=(
                "Students are taught how AND OR and NOT work with truth "
                "values and apply the Boolean operators in several code "
                "examples."
            ),
            expected_decisions=("compatible",),
            expected_safe_reuse=True,
        ),

        case(
            name="add-boolean-negative-incidental",
            edit_action="add_topic",
            source_topic=None,
            source_role=None,
            target_topic="Boolean operations",
            target_role="supporting",
            reviewer_reason=(
                "Boolean operators are explicitly taught but were missed "
                "from the detected topic list."
            ),
            stored_evidence=(
                "The lesson explicitly teaches Boolean AND OR and NOT "
                "operators, including truth conditions and worked examples."
            ),
            current_evidence=(
                "A found flag is set to false and checked in a while loop "
                "while the lesson explains binary search pointer updates."
            ),
            expected_decisions=(
                "incompatible",
                "uncertain",
            ),
            expected_safe_reuse=False,
        ),

        # --------------------------------------------------------------
        # REPLACE TOPIC
        # --------------------------------------------------------------
        case(
            name="replace-subroutine-positive",
            edit_action="replace_topic",
            source_topic="Subroutine statements",
            source_role="supporting",
            target_topic="Subroutines, procedures and functions",
            target_role="supporting",
            reviewer_reason=(
                "The transcript teaches functions and procedures, "
                "parameters, local variables and returned values; the "
                "generic subroutine-statement mapping is too broad."
            ),
            stored_evidence=(
                "The lesson explains functions and procedures, their "
                "parameters, local variables and returned values."
            ),
            current_evidence=(
                "Students learn procedures and functions, including "
                "parameters, return values and local scope."
            ),
            expected_decisions=("compatible",),
            expected_safe_reuse=True,
        ),

        case(
            name="replace-subroutine-negative-general-statements",
            edit_action="replace_topic",
            source_topic="Subroutine statements",
            source_role="supporting",
            target_topic="Subroutines, procedures and functions",
            target_role="supporting",
            reviewer_reason=(
                "The transcript teaches functions and procedures, "
                "parameters, local variables and returned values; the "
                "generic subroutine-statement mapping is too broad."
            ),
            stored_evidence=(
                "The lesson explains functions and procedures, their "
                "parameters, local variables and returned values."
            ),
            current_evidence=(
                "The lesson teaches assignment, selection and iteration "
                "statements as the main programming constructs, using IF "
                "statements and loops."
            ),
            expected_decisions=(
                "incompatible",
                "uncertain",
            ),
            expected_safe_reuse=False,
        ),
    ]

    rows = []
    false_positive_count = 0
    false_negative_count = 0

    for index, diagnostic in enumerate(
        cases,
        start=1,
    ):
        print()
        print("-" * 100)
        print(
            f"CASE {index}/{len(cases)} — "
            f"{diagnostic.name}"
        )
        print("-" * 100)

        try:
            result = validator.validate(
                diagnostic.request
            )
        except Exception as exc:
            print(
                "ERROR:",
                type(exc).__name__,
                str(exc),
            )

            rows.append(
                {
                    "name": diagnostic.name,
                    "decision": "ERROR",
                    "confidence": None,
                    "safe_reuse": False,
                    "expected_safe": (
                        diagnostic.expected_safe_reuse
                    ),
                    "pass": False,
                    "explanation": str(exc),
                }
            )
            continue

        decision_ok = (
            result.decision
            in diagnostic.expected_decisions
        )
        safe_ok = (
            result.safe_for_automatic_reuse
            == diagnostic.expected_safe_reuse
        )

        passed = (
            decision_ok
            and safe_ok
        )

        if (
            result.safe_for_automatic_reuse
            and not diagnostic.expected_safe_reuse
        ):
            false_positive_count += 1

        if (
            not result.safe_for_automatic_reuse
            and diagnostic.expected_safe_reuse
        ):
            false_negative_count += 1

        print(
            "expected decision :",
            " / ".join(
                diagnostic.expected_decisions
            ),
        )
        print(
            "observed decision :",
            result.decision,
        )
        print(
            "confidence        :",
            f"{result.confidence:.3f}",
        )
        print(
            "rationale applies :",
            result.rationale_still_applies,
        )
        print(
            "same context      :",
            result.same_teaching_context,
        )
        print(
            "independent teach :",
            result.independent_teaching_detected,
        )
        print(
            "safe reuse        :",
            result.safe_for_automatic_reuse,
        )
        print(
            "expected safe     :",
            diagnostic.expected_safe_reuse,
        )
        print(
            "explanation       :",
            result.explanation,
        )
        print(
            "result            :",
            "PASS" if passed else "REVIEW",
        )

        rows.append(
            {
                "name": diagnostic.name,
                "decision": result.decision,
                "confidence": result.confidence,
                "safe_reuse": (
                    result.safe_for_automatic_reuse
                ),
                "expected_safe": (
                    diagnostic.expected_safe_reuse
                ),
                "pass": passed,
                "explanation": result.explanation,
            }
        )

    total_passed = sum(
        bool(row["pass"])
        for row in rows
    )

    print()
    print("=" * 100)
    print("STEP 4.7 SUMMARY")
    print("=" * 100)
    print(
        "Cases passed:",
        f"{total_passed}/{len(cases)}",
    )
    print(
        "False-positive automatic reuses:",
        false_positive_count,
    )
    print(
        "False-negative safe reuses:",
        false_negative_count,
    )
    print()
    print(
        "No PostgreSQL rows were written."
    )
    print(
        "No Module 3 / Streamlit / Qdrant / mapping-memory "
        "code was modified."
    )
    print(
        "No edit was applied to a real transcript."
    )
    print()

    # Precision-first readiness:
    # absolutely no false-positive automatic edit is acceptable at this
    # diagnostic stage. False negatives are less dangerous and are reviewed
    # separately.
    if (
        false_positive_count == 0
        and total_passed == len(cases)
    ):
        print(
            "READY_FOR_CONTROLLED_OVERLAY_TEST = YES"
        )
    else:
        print(
            "READY_FOR_CONTROLLED_OVERLAY_TEST = NO"
        )
        print(
            "Inspect every REVIEW/ERROR case before any pipeline wiring."
        )


if __name__ == "__main__":
    main()
