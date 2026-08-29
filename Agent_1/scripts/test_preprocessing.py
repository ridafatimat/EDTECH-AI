from app.services.transcript_preprocessor import preprocess_transcript


def print_result(title: str, transcript: str) -> None:
    """
    Run preprocessing and print the result clearly.
    """

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    print("\nRAW TRANSCRIPT:\n")
    print(transcript)

    result = preprocess_transcript(transcript)

    print("\nCLEANED TRANSCRIPT:\n")
    print(result.cleaned_text)

    print("\nLLM FALLBACK NEEDED:")
    print(result.needs_llm_refinement)

    print("\nFALLBACK REASONS:")
    print(result.fallback_reasons)

    print("\nPREPROCESSING STATS:")
    print(
        result.stats.model_dump_json(
            indent=2
        )
    )


# =========================================================
# TEST CASE 1
# Normal transcript
#
# Expected:
# - timestamps removed
# - speaker labels removed
# - fillers removed
# - artefacts removed
# - technical content preserved
# - LLM fallback = False
# =========================================================

normal_transcript = """
00:01 Teacher: Um, today we are going to learn about binary search.

00:06 Teacher: Uh, binary search only works on a sorted list.

00:15 Teacher: For example, um, mid = (low + high) // 2.

[noise]

00:20 Teacher: If array[mid] == target, return mid.

00:25 Teacher: If array[mid] < target, then low = mid + 1.

00:30 Teacher: Otherwise, high = mid - 1.
"""


# =========================================================
# TEST CASE 2
# Transcript containing uncertain content
#
# Expected:
# - normal cleaning still happens
# - [unclear] and ??? remain
# - LLM fallback = True
# =========================================================

uncertain_transcript = """
00:01 Teacher: Today we will discuss searching algorithms.

00:08 Teacher: Binary search works by checking [unclear] ??? part of the list.

00:15 Teacher: The list must be sorted before binary search can be used.
"""


# =========================================================
# TEST CASE 3
# Code / pseudocode preservation
#
# Expected:
# technical syntax should remain intact
# =========================================================

technical_transcript = """
00:01 Teacher: Um, consider this pseudocode.

00:05 Teacher: mid = (low + high) // 2

00:10 Teacher: if array[mid] == target

00:15 Teacher: return mid

00:20 Teacher: if array[mid] < target

00:25 Teacher: low = mid + 1
"""


if __name__ == "__main__":

    print_result(
        "TEST 1 - NORMAL TRANSCRIPT",
        normal_transcript,
    )

    print_result(
        "TEST 2 - UNCERTAIN TRANSCRIPT",
        uncertain_transcript,
    )

    print_result(
        "TEST 3 - TECHNICAL CONTENT",
        technical_transcript,
    )