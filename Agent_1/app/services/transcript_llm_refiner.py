from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from groq import Groq

from app.schemas.transcript import LLMRefinementResult


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODEL_NAME = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b",
)


# =========================================================
# CONTEXTUAL TRANSCRIPT CLEANING PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are a transcript-cleaning assistant for educational
Computer Science lessons.

The transcript has already undergone deterministic mechanical
cleaning. Timestamps, speaker labels, filler words, obvious
noise markers and explicit uncertainty markers may already
have been removed.

Your job is to return a clean lesson transcript while
preserving the original educational meaning.


============================================================
1. REMOVE IRRELEVANT CLASSROOM CONTENT
============================================================

Remove speech that is clearly unrelated to the academic lesson.

This can include:

- classroom management
- behaviour instructions
- room/environment comments
- equipment problems
- unrelated scheduling information
- unrelated homework/admin comments
- unrelated interruptions
- document/test headings that are not part of the lesson

Use the surrounding lesson context to decide relevance.

Do NOT remove a student question simply because a student
asked it.

Keep questions and answers that contribute to the lesson.


============================================================
2. CORRECT OBVIOUS TRANSCRIPTION / ASR ERRORS
============================================================

Correct obvious spelling or transcription errors when the
intended wording is clear from context.

Examples include obvious misspellings of known Computer
Science terminology.

If the intended wording is uncertain, do NOT guess.

Do not replace uncertain content with invented information.


============================================================
3. HANDLE BROKEN REMNANTS SAFELY
============================================================

Mechanical preprocessing may have removed markers such as:

[unclear]
[unknown]
???
<unk>

This can sometimes leave grammatically incomplete fragments.

When this happens:

- Do NOT invent the missing words.
- If a meaningful part of the sentence can be preserved
  without guessing, keep only that meaningful part.
- If the entire fragment cannot be made meaningful without
  reconstructing missing speech, remove that incomplete
  fragment.
- Never create new lesson content to fill the gap.

Example principle:

Broken:
"The explanation is, but the variable controls the loop."

Safe:
"The variable controls the loop."

Do not guess what was missing before "but".


============================================================
4. PRESERVE LESSON CONTENT
============================================================

Do NOT:

- summarize the lesson
- compress several explanations into one
- paraphrase correct content simply to improve style
- add new definitions
- add new examples
- add facts
- invent missing speech
- change the teacher's intended meaning

Your task is CLEANING, not rewriting.


============================================================
5. PROTECT TECHNICAL CONTENT
============================================================

Preserve:

- Computer Science terminology
- algorithm names
- variable names
- numeric values used inside technical expressions
- code
- pseudocode
- SQL commands
- programming operators
- array/list indexing
- function names
- logical conditions

Examples:

array[mid]
numbers[i]
count += 1
mid = (low + high) // 2
array[mid] == target
array[mid] != target
numbers[i] >= 10

Do NOT:

- rename variables
- change operators
- rewrite code into prose
- change the order of algorithm steps
- change technical values
- reformat code unnecessarily

Preserve code and pseudocode lines as closely as possible to
the input.


============================================================
6. FORMATTING
============================================================

You may:

- fix obvious capitalization
- fix obvious punctuation
- remove accidental spacing
- remove blank fragments

Do not make stylistic changes unless they are needed for the
transcript to remain understandable.


============================================================
7. OUTPUT
============================================================

Return exactly:

1. refined_text
   The cleaned transcript.

2. changes_made
   A short list of meaningful corrections/removals.

Do not report tiny whitespace changes.

If no contextual cleaning is required, return the transcript
unchanged and return an empty changes_made list.
"""


# =========================================================
# GPT-OSS REFINEMENT
# =========================================================

def refine_transcript_with_llm(
    cleaned_text: str,
    fallback_reasons: list[str] | None = None,
) -> LLMRefinementResult:
    """
    Contextually clean a transcript after deterministic
    preprocessing.

    GPT-OSS handles:
        - irrelevant classroom chatter
        - obvious ASR/spelling errors
        - broken remnants left by uncertainty-marker removal

    It must preserve:
        - educational meaning
        - technical terminology
        - code and pseudocode
    """

    if not cleaned_text or not cleaned_text.strip():
        raise ValueError(
            "Cannot refine an empty transcript."
        )

    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY was not found. "
            "Add it to Agent_1/.env."
        )

    client = Groq(
        api_key=GROQ_API_KEY
    )

    reasons = fallback_reasons or []

    user_prompt = f"""
Clean the following Computer Science lesson transcript.

Signals from deterministic preprocessing:

{json.dumps(reasons, indent=2)}

Even when no signals are listed, review the transcript for:

- irrelevant classroom/admin chatter
- obvious transcription or spelling errors
- broken sentence fragments caused by removed uncertainty markers

TRANSCRIPT
------------------------------------------------------------
{cleaned_text}
------------------------------------------------------------

Important:

- preserve all lesson-relevant content
- preserve lesson-relevant student questions
- remove clearly unrelated classroom chatter
- correct only obvious transcription errors
- do not guess missing speech
- remove broken fragments only when they cannot be safely
  preserved without guessing
- preserve technical terminology
- preserve code and pseudocode
- preserve variables and operators
- do not summarize
- do not add information
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,

        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],

        reasoning_effort="low",

        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "transcript_refinement",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "refined_text": {
                            "type": "string"
                        },
                        "changes_made": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        },
                    },
                    "required": [
                        "refined_text",
                        "changes_made",
                    ],
                    "additionalProperties": False,
                },
            },
        },
    )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError(
            "GPT-OSS returned an empty response."
        )

    try:
        parsed_response = json.loads(
            content
        )

    except json.JSONDecodeError as error:
        raise RuntimeError(
            "GPT-OSS returned invalid JSON."
        ) from error

    return LLMRefinementResult.model_validate(
        parsed_response
    )