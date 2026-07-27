from app.services.transcript_llm_refiner import (
    refine_transcript_with_llm,
)


# =========================================================
# TEST TRANSCRIPT
# =========================================================

uncertain_transcript = """
Today we will discuss binary search.

Binary search works by checking [unclear] ??? part of the sorted list.

The list must be sorted before binary search can be used.

mid = (low + high) // 2

if array[mid] == target
    return mid
"""


fallback_reasons = [
    "Transcript contains unresolved transcription markers."
]


# =========================================================
# RUN TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 80)
    print("GPT-OSS TRANSCRIPT REFINEMENT TEST")
    print("=" * 80)

    print("\nINPUT TRANSCRIPT:\n")
    print(uncertain_transcript)

    print("\nSending transcript to GPT-OSS-120B...\n")

    try:
        result = refine_transcript_with_llm(
            cleaned_text=uncertain_transcript,
            fallback_reasons=fallback_reasons,
        )

        print("=" * 80)
        print("REFINED TRANSCRIPT")
        print("=" * 80)

        print(result.refined_text)

        print("\n" + "=" * 80)
        print("CHANGES MADE")
        print("=" * 80)

        if result.changes_made:
            for change in result.changes_made:
                print(f"- {change}")
        else:
            print("No changes made.")

        print("\n" + "=" * 80)
        print("UNRESOLVED SEGMENTS")
        print("=" * 80)

        if result.unresolved_segments:
            for segment in result.unresolved_segments:
                print(f"- {segment}")
        else:
            print("None.")

    except Exception as error:

        print("\nGPT-OSS TEST FAILED")
        print(f"Error: {error}")