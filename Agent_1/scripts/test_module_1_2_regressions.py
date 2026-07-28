from __future__ import annotations

from app.services.semantic_chunker import SemanticChunker
from app.services.transcript_preprocessor import preprocess_transcript


def test_cross_line_sentence_deduplication() -> None:
    raw = (
        "Teacher: Okay.\n"
        "Student: Okay.\n"
        "Teacher: Now we continue."
    )

    result = preprocess_transcript(raw)

    assert result.cleaned_text.count("Okay.") == 1
    assert result.stats.repeated_sentences_removed == 1


def test_affirmation_tokens_are_preserved() -> None:
    raw = (
        "Teacher: uh-huh, that is correct.\n"
        "Student: um-hmm, I understand.\n"
        "Teacher: um, continue please."
    )

    result = preprocess_transcript(raw)

    assert "uh-huh" in result.cleaned_text.lower()
    assert "um-hmm" in result.cleaned_text.lower()
    assert "um, continue" not in result.cleaned_text.lower()
    assert result.stats.fillers_removed == 1


def test_notification_noise_is_removed() -> None:
    raw = (
        "Teacher: Start the question. "
        "[notification noise] "
        "Now continue."
    )

    result = preprocess_transcript(raw)

    assert "[notification noise]" not in result.cleaned_text.lower()
    assert result.stats.artefacts_removed == 1


def test_before_we_move_on_is_not_a_transition() -> None:
    chunker = SemanticChunker()

    assert (
        chunker._transition_strength(
            "Before we move on, one last question about linear search."
        )
        is None
    )


def test_real_transition_still_detected() -> None:
    chunker = SemanticChunker()

    assert (
        chunker._transition_strength(
            "Let's move on to the next question."
        )
        == "strong"
    )


def main() -> None:
    tests = [
        test_cross_line_sentence_deduplication,
        test_affirmation_tokens_are_preserved,
        test_notification_noise_is_removed,
        test_before_we_move_on_is_not_a_transition,
        test_real_transition_still_detected,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print("\nALL MODULE 1 + MODULE 2 REGRESSION TESTS PASSED")


if __name__ == "__main__":
    main()