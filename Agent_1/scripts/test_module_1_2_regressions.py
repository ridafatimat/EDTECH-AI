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




def test_source_filename_heading_is_removed() -> None:
    raw = (
        "Transcript Test 3 - Algorithms and Programming Stress Test\n"
        "Teacher: Today we are learning linear search.\n"
        "Teacher: Linear search checks each item in order."
    )

    result = preprocess_transcript(
        raw,
        source_name=(
            "Transcript_Test_3_Algorithms_Programming_Stress.docx"
        ),
    )

    assert "Transcript Test 3" not in result.cleaned_text
    assert "Today we are learning linear search" in result.cleaned_text
    assert result.stats.metadata_lines_removed == 1
    assert result.removed_metadata_lines == [
        "Transcript Test 3 - Algorithms and Programming Stress Test"
    ]


def test_synthetic_test_label_is_removed() -> None:
    raw = (
        "Synthetic raw lesson transcript for EDTech Module 1 + "
        "Module 2 testing\n"
        "Teacher: Arrays store several values under one name."
    )

    result = preprocess_transcript(raw)

    assert "Synthetic raw lesson transcript" not in result.cleaned_text
    assert "Arrays store several values" in result.cleaned_text
    assert result.stats.metadata_lines_removed == 1


def test_genuine_lesson_introduction_is_preserved() -> None:
    raw = (
        "Teacher: Today we are learning algorithms and programming "
        "techniques.\n"
        "Teacher: Let's begin with linear search."
    )

    result = preprocess_transcript(
        raw,
        source_name="Algorithms_Lesson.docx",
    )

    assert "Today we are learning algorithms" in result.cleaned_text
    assert "Let's begin with linear search" in result.cleaned_text
    assert result.stats.metadata_lines_removed == 0


def test_page_markers_and_repeated_source_headers_are_removed() -> None:
    raw = (
        "Networks Lesson\n"
        "Teacher: A protocol is a set of communication rules.\n"
        "Page 1 of 2\n"
        "Networks Lesson\n"
        "Teacher: TCP confirms delivery.\n"
        "2 / 2"
    )

    result = preprocess_transcript(
        raw,
        source_name="Networks_Lesson.docx",
    )

    assert "Networks Lesson" not in result.cleaned_text
    assert "Page 1 of 2" not in result.cleaned_text
    assert "2 / 2" not in result.cleaned_text
    assert "A protocol is a set of communication rules" in result.cleaned_text
    assert "TCP confirms delivery" in result.cleaned_text
    assert result.stats.metadata_lines_removed == 4


def test_punctuationless_caption_is_split_into_bounded_sentences() -> None:
    chunker = SemanticChunker()

    raw_caption = " ".join(
        [
            "algorithm efficiency compares two solutions that perform the same task",
            "the first solution repeats an instruction inside a loop",
            "the second solution calculates the answer using one expression",
            "when the input becomes larger the loop executes many more times",
            "however the direct calculation still executes once",
            "therefore the second algorithm is more time efficient",
        ]
        * 8
    )

    sentences = chunker._split_sentences(raw_caption)

    assert len(sentences) > 1
    assert max(
        len(sentence.split())
        for sentence in sentences
    ) <= chunker.config.max_sentence_words


def test_punctuationless_caption_does_not_create_giant_semantic_unit() -> None:
    chunker = SemanticChunker()

    raw_caption = " ".join(
        [
            "we compare algorithm efficiency using two functions",
            "one function uses a for loop and executes repeatedly",
            "the other function uses one arithmetic expression",
            "as the input size grows the repeated solution takes longer",
            "the direct calculation remains quick",
        ]
        * 25
    )

    sentences = chunker._split_sentences(raw_caption)
    units = chunker._build_semantic_units(sentences)

    assert len(units) > 2
    assert max(unit.word_count for unit in units) < 150


def test_normal_punctuated_sentences_are_preserved() -> None:
    chunker = SemanticChunker()

    text = (
        "Linear search checks each item in order. "
        "Binary search checks the middle item of sorted data. "
        "Both algorithms solve a searching problem."
    )

    sentences = chunker._split_sentences(text)

    assert sentences == [
        "Linear search checks each item in order.",
        "Binary search checks the middle item of sorted data.",
        "Both algorithms solve a searching problem.",
    ]

def main() -> None:
    tests = [
        test_cross_line_sentence_deduplication,
        test_affirmation_tokens_are_preserved,
        test_notification_noise_is_removed,
        test_source_filename_heading_is_removed,
        test_synthetic_test_label_is_removed,
        test_genuine_lesson_introduction_is_preserved,
        test_page_markers_and_repeated_source_headers_are_removed,
        test_before_we_move_on_is_not_a_transition,
        test_real_transition_still_detected,
        test_punctuationless_caption_is_split_into_bounded_sentences,
        test_punctuationless_caption_does_not_create_giant_semantic_unit,
        test_normal_punctuated_sentences_are_preserved,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print("\nALL MODULE 1 + MODULE 2 REGRESSION TESTS PASSED")


if __name__ == "__main__":
    main()