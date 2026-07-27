from app.services.hybrid_preprocessor import (
    preprocess_transcript_hybrid,
)


def run_test(
    title: str,
    transcript: str,
) -> None:
    """
    Run one transcript through the complete hybrid pipeline
    and print the result.
    """

    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

    print("\nRAW TRANSCRIPT:\n")
    print(transcript)

    try:
        result = preprocess_transcript_hybrid(
            raw_text=transcript
        )

        print("\n" + "=" * 90)
        print("FINAL CLEANED TRANSCRIPT")
        print("=" * 90)

        print(result.cleaned_text)

        print("\nLLM USED:")
        print(result.llm_used)

        print("\nLLM CHANGES:")

        if result.llm_changes:
            for change in result.llm_changes:
                print(f"- {change}")
        else:
            print("None")

        print("\nUNRESOLVED SEGMENTS:")

        if result.unresolved_segments:
            for segment in result.unresolved_segments:
                print(f"- {segment}")
        else:
            print("None")

        print("\nPREPROCESSING STATS:")
        print(
            result.stats.model_dump_json(
                indent=2
            )
        )

    except Exception as error:

        print("\nPIPELINE FAILED")
        print(f"Error: {error}")


# =========================================================
# TEST 1
# Normal transcript
#
# EXPECTED:
# - Python/regex performs cleaning
# - GPT-OSS should NOT be called
# - llm_used should be False
# =========================================================

normal_transcript = """
00:01 Teacher: Um, today we are going to learn about binary search.

00:06 Teacher: Uh, binary search only works on a sorted list.

00:15 Teacher: mid = (low + high) // 2

[noise]

00:20 Teacher: if array[mid] == target

00:25 Teacher: return mid
"""


# =========================================================
# TEST 2
# Uncertain transcript
#
# EXPECTED:
# - Rule-based cleaning happens first
# - uncertainty is detected
# - GPT-OSS is automatically called
# - llm_used should be True
# =========================================================

uncertain_transcript = """
00:01 Teacher: Today we will discuss binary search.

00:08 Teacher: Binary search works by checking [unclear] ??? part of the sorted list.

00:15 Teacher: The list must be sorted before binary search can be used.

00:20 Teacher: mid = (low + high) // 2

00:25 Teacher: if array[mid] == target

00:30 Teacher: return mid
"""


# =========================================================
# TEST 3
# Obvious transcription error
#
# EXPECTED:
# Current rule-based detector may or may not send this to
# the LLM depending on whether an uncertainty trigger exists.
#
# For now, we include an uncertainty marker so the fallback
# is definitely triggered.
# =========================================================

repairable_transcript = """
00:01 Teacher: Today we are learning binary serch.

00:08 Teacher: Binary serch requires the list to be sorted.

00:15 Teacher: [unclear]

00:20 Teacher: mid = (low + high) // 2
"""


if __name__ == "__main__":

    run_test(
        title="TEST 1 - NORMAL TRANSCRIPT",
        transcript=normal_transcript,
    )

    run_test(
        title="TEST 2 - UNCERTAIN TRANSCRIPT",
        transcript=uncertain_transcript,
    )

    run_test(
        title="TEST 3 - LLM REPAIR TEST",
        transcript=repairable_transcript,
    )