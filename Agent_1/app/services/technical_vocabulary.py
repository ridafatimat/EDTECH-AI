from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass


_CORE_TERMS = {
    "algorithm",
    "algorithm tracing",
    "array",
    "array index",
    "array length",
    "binary",
    "binary search",
    "bit",
    "bitmap",
    "Boolean",
    "Boolean logic",
    "bubble sort",
    "byte",
    "bytecode",
    "cache memory",
    "character",
    "class",
    "compiler",
    "constructor",
    "data structure",
    "denary",
    "field",
    "for loop",
    "function",
    "global variable",
    "hexadecimal",
    "index",
    "integer",
    "interpreter",
    "iteration",
    "linear search",
    "local variable",
    "logic error",
    "merge sort",
    "parameter",
    "procedure",
    "program execution",
    "pseudocode",
    "record",
    "recursion",
    "relational operator",
    "return value",
    "runtime error",
    "selection",
    "sequence",
    "sorting algorithm",
    "SQL",
    "statement",
    "string",
    "subroutine",
    "syntax error",
    "trace table",
    "validation",
    "variable",
    "verification",
    "while loop",
}

_TECHNICAL_CONTEXT_ANCHORS = {
    "algorithm",
    "array",
    "binary",
    "bit",
    "boolean",
    "byte",
    "cache",
    "class",
    "code",
    "compiler",
    "condition",
    "constructor",
    "cpu",
    "data",
    "database",
    "debug",
    "error",
    "execute",
    "function",
    "hexadecimal",
    "index",
    "instruction",
    "integer",
    "iteration",
    "loop",
    "memory",
    "parameter",
    "procedure",
    "program",
    "pseudocode",
    "query",
    "record",
    "recursion",
    "search",
    "sort",
    "statement",
    "string",
    "subroutine",
    "table",
    "validation",
    "variable",
    "verification",
}

# A single occurrence of these words is not enough to prove that a sentence
# is technical. Two weak anchors, or one strong anchor, are required.
_WEAK_CONTEXT_ANCHORS = {
    "bit",
    "class",
    "error",
    "field",
    "function",
    "loop",
    "memory",
    "record",
    "string",
}

# These are valid high-frequency grammatical/control words. A fuzzy matcher
# must not reinterpret one of them as a technical term merely because the
# surrounding sentence is technical.
_PROTECTED_TOKENS = {
    "a",
    "an",
    "and",
    "another",
    "any",
    "as",
    "at",
    "both",
    "by",
    "do",
    "each",
    "else",
    "every",
    "false",
    "for",
    "from",
    "if",
    "in",
    "is",
    "no",
    "not",
    "of",
    "on",
    "one",
    "or",
    "same",
    "so",
    "than",
    "then",
    "to",
    "true",
    "until",
    "when",
    "where",
    "while",
    "with",
    "yes",
}


@dataclass(frozen=True, slots=True)
class TechnicalVocabulary:
    canonical_terms: tuple[str, ...]
    context_anchors: frozenset[str]
    weak_context_anchors: frozenset[str]
    protected_tokens: frozenset[str]

    @property
    def canonical_term_keys(self) -> frozenset[str]:
        return frozenset(
            normalise_lookup_key(term)
            for term in self.canonical_terms
        )

    @property
    def canonical_records(
        self,
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        records: list[tuple[str, tuple[str, ...]]] = []

        for term in self.canonical_terms:
            tokens = tokenise_lookup_key(term)

            if tokens:
                records.append((term, tokens))

        return tuple(records)

    def is_valid_surface_form(self, value: str) -> bool:
        """
        Return True for an exact canonical term or a harmless grammatical
        surface form such as:

        - while loop / while loops
        - subroutine / subroutines
        - return value / returns value

        These forms are valid transcript wording and must not be "corrected"
        into a different number or verb form.
        """

        tokens = tokenise_lookup_key(value)

        if not tokens:
            return False

        for _, canonical_tokens in self.canonical_records:
            if len(tokens) != len(canonical_tokens):
                continue

            if all(
                left == right
                or are_simple_inflectional_variants(left, right)
                for left, right in zip(tokens, canonical_tokens)
            ):
                return True

        return False


def normalise_lookup_key(value: str) -> str:
    """
    Build a stable lookup key without changing the original transcript.

    Examples:
    - "While-Loop" -> "while loop"
    - "array.length" -> "array length"
    """

    value = value.casefold()
    value = re.sub(r"[_./\\-]+", " ", value)
    value = re.sub(r"[^a-z0-9\s]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def tokenise_lookup_key(value: str) -> tuple[str, ...]:
    key = normalise_lookup_key(value)

    if not key:
        return ()

    return tuple(key.split())


def are_simple_inflectional_variants(
    left: str,
    right: str,
) -> bool:
    """
    Detect only conservative English s/es/ies surface variants.

    This intentionally covers the false positives observed in transcripts,
    while avoiding broad stemming that could collapse unrelated words.
    """

    left = left.casefold()
    right = right.casefold()

    if left == right:
        return True

    shorter, longer = sorted(
        (left, right),
        key=len,
    )

    if len(shorter) < 3:
        return False

    if longer == shorter + "s":
        return True

    if longer == shorter + "es":
        return True

    if (
        shorter.endswith("y")
        and longer == shorter[:-1] + "ies"
    ):
        return True

    return False


def build_technical_vocabulary(
    *,
    extra_terms: Iterable[str] = (),
    extra_protected_tokens: Iterable[str] = (),
) -> TechnicalVocabulary:
    """
    Build the controlled vocabulary from:
    1. core Computer Science terms;
    2. PostgreSQL syllabus concepts through SyllabusStore;
    3. caller-provided terms.

    PostgreSQL is now the authoritative syllabus source. The old static catalogue module is intentionally not imported here.
    No production correction mapping is stored in this vocabulary; it only
    supplies canonical terminology and precision safeguards.
    """

    terms = set(_CORE_TERMS)
    terms.update(_load_terms_from_syllabus_store())
    terms.update(
        term.strip()
        for term in extra_terms
        if isinstance(term, str) and term.strip()
    )

    by_key: dict[str, str] = {}

    for term in terms:
        key = normalise_lookup_key(term)

        if not key:
            continue

        existing = by_key.get(key)

        # Prefer the better-formatted or more informative representation.
        if existing is None or len(term) > len(existing):
            by_key[key] = term

    canonical_terms = tuple(
        sorted(
            by_key.values(),
            key=lambda value: (
                len(value.split()),
                value.casefold(),
            ),
        )
    )

    protected_tokens = set(_PROTECTED_TOKENS)
    protected_tokens.update(
        normalise_lookup_key(token)
        for token in extra_protected_tokens
        if isinstance(token, str)
        and normalise_lookup_key(token)
    )

    return TechnicalVocabulary(
        canonical_terms=canonical_terms,
        context_anchors=frozenset(_TECHNICAL_CONTEXT_ANCHORS),
        weak_context_anchors=frozenset(_WEAK_CONTEXT_ANCHORS),
        protected_tokens=frozenset(protected_tokens),
    )


def _load_terms_from_syllabus_store() -> set[str]:
    """
    Load canonical syllabus terminology from PostgreSQL via SyllabusStore.

    The import is deliberately local so importing ``technical_vocabulary``
    itself does not create a syllabus-store dependency until the vocabulary
    is actually built. Qdrant is not queried by this function.

    A database/storage error is allowed to propagate. Silently falling back
    to only ``_CORE_TERMS`` would hide an incomplete syllabus lookup after
    removal of the static catalogue.
    """

    from app.services.syllabus_store import SyllabusStore

    store = SyllabusStore()
    return set(store.get_technical_terms())
