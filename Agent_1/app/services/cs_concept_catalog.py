from __future__ import annotations

from collections import defaultdict
import re
from dataclasses import dataclass
from typing import Literal


Paper = Literal["Paper 1", "Paper 2"]


@dataclass(frozen=True)
class FlexiblePattern:
    """
    Reusable flexible lexical pattern for natural classroom language.

    Exact aliases are ideal when the transcript contains stable technical
    wording. Patterns handle grammatical variation, passive voice and short
    inserted tokens without tying the extractor to one transcript.
    """

    label: str
    regex: str
    weight: float = 0.82


@dataclass(frozen=True)
class CSConcept:
    """
    One searchable concept derived from the official AQA GCSE
    Computer Science (8525) specification.

    Important:
    - official_reference, chapter_title and official_title come from
      the AQA specification structure.
    - label and description are concise retrieval-friendly summaries.
    - aliases are transcript-friendly matching terms. They are not
      presented as official AQA wording.
    - Several searchable concepts may share one official reference
      when AQA groups multiple assessable ideas under one section.
    """

    concept_id: str

    # Official AQA hierarchy
    official_reference: str
    chapter_reference: str
    chapter_title: str
    official_title: str

    # Retrieval-friendly fields used by Module 3
    label: str
    domain: str
    description: str
    aliases: tuple[str, ...]

    paper: Paper
    source_pages: tuple[int, ...]

    parent_concept_id: str | None = None

    # Phrases that contain an alias but represent a different concept.
    # This metadata is reusable for any catalogue entry and prevents a
    # shorter term from consuming a longer compound term.
    excluded_phrases: tuple[str, ...] = ()

    # Flexible regex patterns for concepts whose classroom wording can vary.
    # Patterns are evaluated against normalised sentence text.
    match_patterns: tuple[FlexiblePattern, ...] = ()

    @property
    def embedding_text(self) -> str:
        """
        Text that can be embedded for semantic topic retrieval.
        """

        aliases = ", ".join(self.aliases)

        return (
            f"AQA GCSE Computer Science {self.official_reference}. "
            f"Chapter: {self.chapter_title}. "
            f"Official topic: {self.official_title}. "
            f"Search concept: {self.label}. "
            f"{self.description} "
            f"Related transcript terms: {aliases}."
        )


def _concept(
    *,
    concept_id: str,
    official_reference: str,
    chapter_reference: str,
    chapter_title: str,
    official_title: str,
    label: str,
    description: str,
    aliases: tuple[str, ...],
    paper: Paper,
    source_pages: tuple[int, ...],
    parent_concept_id: str | None = None,
    excluded_phrases: tuple[str, ...] = (),
    match_patterns: tuple[FlexiblePattern, ...] = (),
) -> CSConcept:
    """
    Small constructor that keeps domain aligned with the official
    AQA chapter title.
    """

    return CSConcept(
        concept_id=concept_id,
        official_reference=official_reference,
        chapter_reference=chapter_reference,
        chapter_title=chapter_title,
        official_title=official_title,
        label=label,
        domain=chapter_title,
        description=description,
        aliases=aliases,
        paper=paper,
        source_pages=source_pages,
        parent_concept_id=parent_concept_id,
        excluded_phrases=excluded_phrases,
        match_patterns=match_patterns,
    )


# =============================================================================
# OFFICIAL AQA CHAPTERS
# =============================================================================

AQA_CHAPTERS: dict[str, str] = {
    "3.1": "Fundamentals of algorithms",
    "3.2": "Programming",
    "3.3": "Fundamentals of data representation",
    "3.4": "Computer systems",
    "3.5": "Fundamentals of computer networks",
    "3.6": "Cyber security",
    "3.7": (
        "Relational databases and structured query language (SQL)"
    ),
    "3.8": (
        "Ethical, legal and environmental impacts of digital "
        "technology on wider society, including issues of privacy"
    ),
}


# =============================================================================
# SEARCHABLE OFFICIAL SYLLABUS CATALOGUE
# =============================================================================

CS_CONCEPTS: tuple[CSConcept, ...] = (
    # =========================================================================
    # 3.1 FUNDAMENTALS OF ALGORITHMS — PAPER 1
    # =========================================================================
    _concept(
        concept_id="aqa_3_1_1_algorithm",
        official_reference="3.1.1",
        chapter_reference="3.1",
        chapter_title=AQA_CHAPTERS["3.1"],
        official_title="Representing algorithms",
        label="Algorithms",
        description=(
            "Understand an algorithm as a sequence of steps and distinguish "
            "an algorithm from its implementation as a computer program."
        ),
        aliases=(
            "algorithm",
            "sequence of steps",
            "solve a task",
            "computer program implementation",
        ),
        paper="Paper 1",
        source_pages=(10,),
    ),
    _concept(
        concept_id="aqa_3_1_1_decomposition",
        official_reference="3.1.1",
        chapter_reference="3.1",
        chapter_title=AQA_CHAPTERS["3.1"],
        official_title="Representing algorithms",
        label="Decomposition",
        description=(
            "Break a problem into smaller sub-problems that each perform "
            "an identifiable task."
        ),
        aliases=(
            "decomposition",
            "break the problem down",
            "split into sub problems",
            "smaller subproblems",
        ),
        paper="Paper 1",
        source_pages=(10,),
        parent_concept_id="aqa_3_1_1_algorithm",
    ),
    _concept(
        concept_id="aqa_3_1_1_abstraction",
        official_reference="3.1.1",
        chapter_reference="3.1",
        chapter_title=AQA_CHAPTERS["3.1"],
        official_title="Representing algorithms",
        label="Abstraction",
        description=(
            "Remove unnecessary detail from a problem so attention remains "
            "on the information required for a solution."
        ),
        aliases=(
            "abstraction",
            "remove unnecessary detail",
            "ignore irrelevant detail",
            "focus on important information",
        ),
        paper="Paper 1",
        source_pages=(10,),
        parent_concept_id="aqa_3_1_1_algorithm",
    ),
    _concept(
        concept_id="aqa_3_1_1_algorithm_representation",
        official_reference="3.1.1",
        chapter_reference="3.1",
        chapter_title=AQA_CHAPTERS["3.1"],
        official_title="Representing algorithms",
        label="Pseudocode, program code and flowcharts",
        description=(
            "Create and represent algorithms systematically using "
            "pseudocode, program code and flowcharts."
        ),
        aliases=(
            "pseudocode",
            "pseudo code",
            "flowchart",
            "program code",
            "represent the algorithm",
            "algorithm design",
        ),
        paper="Paper 1",
        source_pages=(10,),
        parent_concept_id="aqa_3_1_1_algorithm",
    ),
    _concept(
        concept_id="aqa_3_1_1_algorithm_purpose_trace",
        official_reference="3.1.1",
        chapter_reference="3.1",
        chapter_title=AQA_CHAPTERS["3.1"],
        official_title="Representing algorithms",
        label="Algorithm tracing and program execution",
        description=(
            "Identify inputs, processing and outputs, then trace program or "
            "algorithm execution using trace tables, visual inspection and "
            "step-by-step value tracking to determine behaviour or purpose."
        ),
        aliases=(
            "input processing output",
            "inputs processing outputs",
            "purpose of the algorithm",
            "trace table",
            "visual inspection",
            "dry run",
            "dry run the program",
            "trace the algorithm",
            "trace the code",
            "follow the code step by step",
            "follow program execution",
            "track variable values",
            "count statement executions",
            "number of times a statement executes",
            "how many times the loop runs",
            "how many times the loop is executed",
            "statement execution count",
        ),
        match_patterns=(
            FlexiblePattern(
                label="counting statement or loop executions",
                regex=(
                    r"\b(?:how many|number of)\s+times\b"
                    r".{0,60}\b(?:statement|instruction|line|loop)\b"
                    r".{0,60}\b(?:run|runs|ran|running|execute|"
                    r"executes|executed|done)\b"
                ),
                weight=0.84,
            ),
            FlexiblePattern(
                label="reported execution count",
                regex=(
                    r"\b(?:statement|instruction|line|loop)\b"
                    r".{0,50}\b(?:run|runs|ran|running|execute|"
                    r"executes|executed|done)\b"
                    r".{0,35}\b(?:once|twice|[a-z]+\s+times|"
                    r"\d+\s+times)\b"
                ),
                weight=0.82,
            ),
            FlexiblePattern(
                label="tracking changing program values",
                regex=(
                    r"\b(?:value|variable|index|counter)\b"
                    r".{0,45}\b(?:becomes|changes|changed|updated|"
                    r"increases|decreases)\b"
                ),
                weight=0.78,
            ),
            FlexiblePattern(
                label="reasoning about possible execution behaviour",
                regex=(
                    r"\b(?:possible|impossible)\b"
                    r".{0,60}\b(?:statement|instruction|loop|"
                    r"execution|run)\b"
                ),
                weight=0.78,
            ),
        ),
        paper="Paper 1",
        source_pages=(10,),
        parent_concept_id="aqa_3_1_1_algorithm",
    ),
    _concept(
        concept_id="aqa_3_1_2_efficiency",
        official_reference="3.1.2",
        chapter_reference="3.1",
        chapter_title=AQA_CHAPTERS["3.1"],
        official_title="Efficiency of algorithms",
        label="Time efficiency of algorithms",
        description=(
            "Compare algorithms that solve the same problem and explain why "
            "one may be more time-efficient than another."
        ),
        aliases=(
            "algorithm efficiency",
            "time efficiency",
            "more efficient algorithm",
            "compare efficiency",
            "faster algorithm",
            "same problem",
        ),
        paper="Paper 1",
        source_pages=(10,),
    ),
    _concept(
        concept_id="aqa_3_1_3_linear_search",
        official_reference="3.1.3",
        chapter_reference="3.1",
        chapter_title=AQA_CHAPTERS["3.1"],
        official_title="Searching algorithms",
        label="Linear search",
        description=(
            "Understand and explain the mechanics of the linear search "
            "algorithm."
        ),
        aliases=(
            "linear search",
            "sequential search",
            "check each item",
            "search one by one",
            "first item to last item",
        ),
        paper="Paper 1",
        source_pages=(11,),
    ),
    _concept(
        concept_id="aqa_3_1_3_binary_search",
        official_reference="3.1.3",
        chapter_reference="3.1",
        chapter_title=AQA_CHAPTERS["3.1"],
        official_title="Searching algorithms",
        label="Binary search",
        description=(
            "Understand and explain the mechanics of binary search on "
            "ordered data."
        ),
        aliases=(
            "binary search",
            "middle item",
            "middle value",
            "discard half",
            "search sorted data",
            "lower half",
            "upper half",
        ),
        paper="Paper 1",
        source_pages=(11,),
    ),
    _concept(
        concept_id="aqa_3_1_3_search_comparison",
        official_reference="3.1.3",
        chapter_reference="3.1",
        chapter_title=AQA_CHAPTERS["3.1"],
        official_title="Searching algorithms",
        label="Comparing linear and binary search",
        description=(
            "Compare the advantages and disadvantages of linear and binary "
            "search algorithms."
        ),
        aliases=(
            "compare linear and binary search",
            "linear versus binary search",
            "advantages of binary search",
            "disadvantages of linear search",
            "search algorithm comparison",
        ),
        paper="Paper 1",
        source_pages=(11,),
    ),
    _concept(
        concept_id="aqa_3_1_4_sorting_algorithms",
        official_reference="3.1.4",
        chapter_reference="3.1",
        chapter_title=AQA_CHAPTERS["3.1"],
        official_title="Sorting algorithms",
        label="Sorting algorithms",
        description=(
            "Understand sorting as arranging data into an order and use "
            "the compare-and-swap idea when explaining sorting methods."
        ),
        aliases=(
            "sorting algorithm",
            "sorting algorithms",
            "sort the values",
            "arrange values in order",
            "compare and swap",
            "sorting by swapping",
        ),
        paper="Paper 1",
        source_pages=(11,),
    ),
    _concept(
        concept_id="aqa_3_1_4_merge_sort",
        official_reference="3.1.4",
        chapter_reference="3.1",
        chapter_title=AQA_CHAPTERS["3.1"],
        official_title="Sorting algorithms",
        label="Merge sort",
        description=(
            "Understand and explain the mechanics of the merge sort "
            "algorithm."
        ),
        aliases=(
            "merge sort",
            "split the list",
            "merge sorted lists",
            "divide and merge",
            "divide and conquer sort",
        ),
        paper="Paper 1",
        source_pages=(11,),
        parent_concept_id="aqa_3_1_4_sorting_algorithms",
    ),
    _concept(
        concept_id="aqa_3_1_4_bubble_sort",
        official_reference="3.1.4",
        chapter_reference="3.1",
        chapter_title=AQA_CHAPTERS["3.1"],
        official_title="Sorting algorithms",
        label="Bubble sort",
        description=(
            "Understand and explain the mechanics of the bubble sort "
            "algorithm."
        ),
        aliases=(
            "bubble sort",
            "compare adjacent values",
            "adjacent items",
            "swap adjacent",
            "sorting pass",
            "no swaps",
        ),
        paper="Paper 1",
        source_pages=(11,),
        parent_concept_id="aqa_3_1_4_sorting_algorithms",
    ),
    _concept(
        concept_id="aqa_3_1_4_sort_comparison",
        official_reference="3.1.4",
        chapter_reference="3.1",
        chapter_title=AQA_CHAPTERS["3.1"],
        official_title="Sorting algorithms",
        label="Comparing merge sort and bubble sort",
        description=(
            "Compare the advantages and disadvantages of merge sort and "
            "bubble sort."
        ),
        aliases=(
            "compare merge and bubble sort",
            "merge sort versus bubble sort",
            "sorting algorithm comparison",
            "advantages of merge sort",
            "disadvantages of bubble sort",
        ),
        paper="Paper 1",
        source_pages=(11,),
        parent_concept_id="aqa_3_1_4_sorting_algorithms",
    ),

    # =========================================================================
    # 3.2 PROGRAMMING — PAPER 1
    # =========================================================================
    _concept(
        concept_id="aqa_3_2_1_data_types",
        official_reference="3.2.1",
        chapter_reference="3.2",
        chapter_title=AQA_CHAPTERS["3.2"],
        official_title="Data types",
        label="Data types",
        description=(
            "Understand and use integer, real, Boolean, character and "
            "string data types appropriately."
        ),
        aliases=(
            "data type",
            "integer",
            "real number",
            "float",
            "boolean",
            "character",
            "string",
        ),
        paper="Paper 1",
        source_pages=(11, 12),
    ),
    _concept(
        concept_id="aqa_3_2_2_variables_constants_assignment",
        official_reference="3.2.2",
        chapter_reference="3.2",
        chapter_title=AQA_CHAPTERS["3.2"],
        official_title="Programming concepts",
        label="Variables, constants and assignment",
        description=(
            "Use variable declarations, constant declarations and "
            "assignment statements, and understand why named variables and "
            "constants are used."
        ),
        aliases=(
            "variable declaration",
            "constant declaration",
            "assignment statement",
            "assign a value",
            "named constant",
            "variable value",
        ),
        paper="Paper 1",
        source_pages=(12,),
    ),
    _concept(
        concept_id="aqa_3_2_2_iteration",
        official_reference="3.2.2",
        chapter_reference="3.2",
        chapter_title=AQA_CHAPTERS["3.2"],
        official_title="Programming concepts",
        label="Iteration",
        description=(
            "Use definite count-controlled and indefinite "
            "condition-controlled iteration, including conditions at the "
            "start or end of a loop."
        ),
        aliases=(
            "iteration",
            "repetition",
            "for loop",
            "while loop",
            "repeat until",
            "do while",
            "count controlled loop",
            "condition controlled loop",
            "loop condition",
        ),
        paper="Paper 1",
        source_pages=(12, 13),
    ),
    _concept(
        concept_id="aqa_3_2_2_selection",
        official_reference="3.2.2",
        chapter_reference="3.2",
        chapter_title=AQA_CHAPTERS["3.2"],
        official_title="Programming concepts",
        label="Selection",
        description=(
            "Use selection statements to choose which instructions execute."
        ),
        aliases=(
            "selection",
            "if statement",
            "else statement",
            "conditional statement",
            "if condition",
            "choice",
        ),
        paper="Paper 1",
        source_pages=(12, 13),
    ),
    _concept(
        concept_id="aqa_3_2_2_subroutine_statement",
        official_reference="3.2.2",
        chapter_reference="3.2",
        chapter_title=AQA_CHAPTERS["3.2"],
        official_title="Programming concepts",
        label="Subroutine statements",
        description=(
            "Use procedures or functions as statement types within "
            "programs."
        ),
        aliases=(
            "subroutine",
            "procedure",
            "function",
            "method call",
            "call the function",
        ),
        paper="Paper 1",
        source_pages=(12,),
        parent_concept_id="aqa_3_2_10_subroutines",
    ),
    _concept(
        concept_id="aqa_3_2_2_nested_structures",
        official_reference="3.2.2",
        chapter_reference="3.2",
        chapter_title=AQA_CHAPTERS["3.2"],
        official_title="Programming concepts",
        label="Nested selection and iteration",
        description=(
            "Use selection inside selection and iteration inside iteration "
            "or other control structures."
        ),
        aliases=(
            "nested selection",
            "nested if",
            "nested iteration",
            "nested loop",
            "loop inside a loop",
            "if inside if",
        ),
        paper="Paper 1",
        source_pages=(13,),
    ),
    _concept(
        concept_id="aqa_3_2_2_identifiers",
        official_reference="3.2.2",
        chapter_reference="3.2",
        chapter_title=AQA_CHAPTERS["3.2"],
        official_title="Programming concepts",
        label="Meaningful identifier names",
        description=(
            "Use meaningful names for variables, constants and "
            "subroutines, and explain why they improve programs."
        ),
        aliases=(
            "identifier name",
            "meaningful variable name",
            "meaningful identifier",
            "constant name",
            "subroutine name",
            "descriptive name",
        ),
        paper="Paper 1",
        source_pages=(13,),
    ),
    _concept(
        concept_id="aqa_3_2_3_arithmetic_operations",
        official_reference="3.2.3",
        chapter_reference="3.2",
        chapter_title=AQA_CHAPTERS["3.2"],
        official_title="Arithmetic operations in a programming language",
        label="Arithmetic operations",
        description=(
            "Use addition, subtraction, multiplication, real division, "
            "integer division and remainders."
        ),
        aliases=(
            "arithmetic operation",
            "addition",
            "subtraction",
            "multiplication",
            "real division",
            "integer division",
            "div operator",
            "mod operator",
            "remainder",
            "modulo",
        ),
        paper="Paper 1",
        source_pages=(14,),
    ),
    _concept(
        concept_id="aqa_3_2_4_relational_operations",
        official_reference="3.2.4",
        chapter_reference="3.2",
        chapter_title=AQA_CHAPTERS["3.2"],
        official_title="Relational operations in a programming language",
        label="Relational operators",
        description=(
            "Use and interpret equality, inequality and ordered comparison "
            "operators in algorithms and programs."
        ),
        aliases=(
            "relational operator",
            "equal to",
            "not equal to",
            "less than",
            "greater than",
            "less than or equal",
            "greater than or equal",
            "comparison operator",
        ),
        paper="Paper 1",
        source_pages=(14,),
    ),
    _concept(
        concept_id="aqa_3_2_5_boolean_operations",
        official_reference="3.2.5",
        chapter_reference="3.2",
        chapter_title=AQA_CHAPTERS["3.2"],
        official_title="Boolean operations in a programming language",
        label="Boolean operations in programming",
        description=(
            "Use NOT, AND and OR, including combinations of these "
            "operations in iteration and selection conditions."
        ),
        aliases=(
            "boolean operation",
            "and operator",
            "or operator",
            "not operator",
            "boolean condition",
            "both conditions",
            "at least one condition",
        ),
        paper="Paper 1",
        source_pages=(14,),
    ),
    _concept(
        concept_id="aqa_3_2_6_data_structures",
        official_reference="3.2.6",
        chapter_reference="3.2",
        chapter_title=AQA_CHAPTERS["3.2"],
        official_title="Data structures",
        label="Data structures",
        description=(
            "Understand data structures and use suitable structures when "
            "designing solutions."
        ),
        aliases=(
            "data structure",
            "store multiple values",
            "organise data",
            "structured data",
        ),
        paper="Paper 1",
        source_pages=(14, 15),
    ),
    _concept(
        concept_id="aqa_3_2_6_arrays",
        official_reference="3.2.6",
        chapter_reference="3.2",
        chapter_title=AQA_CHAPTERS["3.2"],
        official_title="Data structures",
        label="One- and two-dimensional arrays",
        description=(
            "Use one-dimensional and two-dimensional arrays, or equivalent "
            "structures, to solve simple problems."
        ),
        aliases=(
            "array",
            "arrays",
            "one dimensional array",
            "1d array",
            "two dimensional array",
            "2d array",
            "array index",
            "array length",
            "array traversal",
            "traverse the array",
        ),
        excluded_phrases=(
            "array list",
            "array lists",
            "arraylist",
            "arraylists",
            "dynamic array",
            "resizable array",
        ),
        paper="Paper 1",
        source_pages=(15,),
        parent_concept_id="aqa_3_2_6_data_structures",
    ),
    _concept(
        concept_id="aqa_3_2_6_records",
        official_reference="3.2.6",
        chapter_reference="3.2",
        chapter_title=AQA_CHAPTERS["3.2"],
        official_title="Data structures",
        label="Records",
        description=(
            "Use records, or equivalent structures, in solutions to "
            "simple problems."
        ),
        aliases=(
            "record",
            "record definition",
            "structured record",
            "fields in a record",
            "record data structure",
        ),
        paper="Paper 1",
        source_pages=(15,),
        parent_concept_id="aqa_3_2_6_data_structures",
    ),
    _concept(
        concept_id="aqa_3_2_7_input_output",
        official_reference="3.2.7",
        chapter_reference="3.2",
        chapter_title=AQA_CHAPTERS["3.2"],
        official_title="Input/output",
        label="Program input and output",
        description=(
            "Obtain user input from the keyboard and output program data "
            "or information to the display."
        ),
        aliases=(
            "user input",
            "keyboard input",
            "input statement",
            "output statement",
            "display output",
            "print output",
        ),
        paper="Paper 1",
        source_pages=(15,),
    ),
    _concept(
        concept_id="aqa_3_2_8_string_handling",
        official_reference="3.2.8",
        chapter_reference="3.2",
        chapter_title=AQA_CHAPTERS["3.2"],
        official_title="String handling operations in a programming language",
        label="String handling",
        description=(
            "Use length, position, substring, concatenation, character-code "
            "conversion and string-number conversion operations."
        ),
        aliases=(
            "string handling",
            "string length",
            "substring",
            "concatenation",
            "character code",
            "ascii code conversion",
            "string to integer",
            "integer to string",
            "string conversion",
        ),
        paper="Paper 1",
        source_pages=(15,),
    ),
    _concept(
        concept_id="aqa_3_2_9_random_numbers",
        official_reference="3.2.9",
        chapter_reference="3.2",
        chapter_title=AQA_CHAPTERS["3.2"],
        official_title="Random number generation in a programming language",
        label="Random number generation",
        description=(
            "Use random number generation within computer programs."
        ),
        aliases=(
            "random number",
            "random number generation",
            "generate a random value",
            "random integer",
            "random function",
        ),
        paper="Paper 1",
        source_pages=(15,),
    ),
    _concept(
        concept_id="aqa_3_2_10_subroutines",
        official_reference="3.2.10",
        chapter_reference="3.2",
        chapter_title=AQA_CHAPTERS["3.2"],
        official_title=(
            "Structured programming and subroutines "
            "(procedures and functions)"
        ),
        label="Subroutines, procedures and functions",
        description=(
            "Understand named reusable blocks of code and explain the "
            "advantages of using subroutines."
        ),
        aliases=(
            "subroutine",
            "procedure",
            "function",
            "named block of code",
            "reusable code",
            "call the subroutine",
            "advantages of subroutines",
        ),
        paper="Paper 1",
        source_pages=(16,),
    ),
    _concept(
        concept_id="aqa_3_2_10_parameters_returns",
        official_reference="3.2.10",
        chapter_reference="3.2",
        chapter_title=AQA_CHAPTERS["3.2"],
        official_title=(
            "Structured programming and subroutines "
            "(procedures and functions)"
        ),
        label="Parameters and return values",
        description=(
            "Pass data into subroutines using parameters and pass data out "
            "using return values."
        ),
        aliases=(
            "parameter",
            "parameters",
            "argument",
            "arguments",
            "pass data to a function",
            "return value",
            "calling routine",
        ),
        paper="Paper 1",
        source_pages=(16,),
        parent_concept_id="aqa_3_2_10_subroutines",
    ),
    _concept(
        concept_id="aqa_3_2_10_local_variables",
        official_reference="3.2.10",
        chapter_reference="3.2",
        chapter_title=AQA_CHAPTERS["3.2"],
        official_title=(
            "Structured programming and subroutines "
            "(procedures and functions)"
        ),
        label="Local variables",
        description=(
            "Use local variables, understand their lifetime and scope, and "
            "explain why local variables are good practice."
        ),
        aliases=(
            "local variable",
            "variable scope",
            "only accessible in the function",
            "only exists during the subroutine",
            "local scope",
        ),
        paper="Paper 1",
        source_pages=(16,),
        parent_concept_id="aqa_3_2_10_subroutines",
    ),
    _concept(
        concept_id="aqa_3_2_10_structured_programming",
        official_reference="3.2.10",
        chapter_reference="3.2",
        chapter_title=AQA_CHAPTERS["3.2"],
        official_title=(
            "Structured programming and subroutines "
            "(procedures and functions)"
        ),
        label="Structured and modular programming",
        description=(
            "Describe structured programming with modular components, "
            "well-documented interfaces, local variables, parameters and "
            "return values."
        ),
        aliases=(
            "structured programming",
            "modular programming",
            "modularised program",
            "program modules",
            "documented interface",
            "advantages of structured programming",
        ),
        paper="Paper 1",
        source_pages=(16,),
    ),
    _concept(
        concept_id="aqa_3_2_11_validation",
        official_reference="3.2.11",
        chapter_reference="3.2",
        chapter_title=AQA_CHAPTERS["3.2"],
        official_title="Robust and secure programming",
        label="Data validation routines",
        description=(
            "Write validation routines that check input length, presence "
            "and whether values lie within an allowed range."
        ),
        aliases=(
            "data validation",
            "input validation",
            "range check",
            "length check",
            "presence check",
            "empty string check",
            "valid input",
        ),
        paper="Paper 1",
        source_pages=(17,),
    ),
    _concept(
        concept_id="aqa_3_2_11_authentication",
        official_reference="3.2.11",
        chapter_reference="3.2",
        chapter_title=AQA_CHAPTERS["3.2"],
        official_title="Robust and secure programming",
        label="Simple authentication routines",
        description=(
            "Write simple username-and-password authentication routines."
        ),
        aliases=(
            "authentication routine",
            "username and password",
            "login routine",
            "check username",
            "check password",
            "authenticate the user",
        ),
        paper="Paper 1",
        source_pages=(17,),
    ),
    _concept(
        concept_id="aqa_3_2_11_testing_test_data",
        official_reference="3.2.11",
        chapter_reference="3.2",
        chapter_title=AQA_CHAPTERS["3.2"],
        official_title="Robust and secure programming",
        label="Program testing and test data",
        description=(
            "Understand testing and select, justify and use normal, "
            "boundary and erroneous test data."
        ),
        aliases=(
            "program testing",
            "test data",
            "normal data",
            "typical data",
            "boundary data",
            "extreme data",
            "erroneous data",
            "invalid data",
            "test case",
        ),
        paper="Paper 1",
        source_pages=(17,),
    ),
    _concept(
        concept_id="aqa_3_2_11_errors_debugging",
        official_reference="3.2.11",
        chapter_reference="3.2",
        chapter_title=AQA_CHAPTERS["3.2"],
        official_title="Robust and secure programming",
        label="Syntax errors, logic errors and debugging",
        description=(
            "Correct errors and identify or categorise syntax and logic "
            "errors in algorithms and programs."
        ),
        aliases=(
            "syntax error",
            "logic error",
            "debugging",
            "debug the program",
            "correct the error",
            "find the error",
            "categorise the error",
        ),
        paper="Paper 1",
        source_pages=(17,),
    ),

    # =========================================================================
    # 3.3 FUNDAMENTALS OF DATA REPRESENTATION — PAPER 2
    # =========================================================================
    _concept(
        concept_id="aqa_3_3_1_number_bases",
        official_reference="3.3.1",
        chapter_reference="3.3",
        chapter_title=AQA_CHAPTERS["3.3"],
        official_title="Number bases",
        label="Decimal, binary and hexadecimal number bases",
        description=(
            "Understand decimal base 10, binary base 2 and hexadecimal "
            "base 16."
        ),
        aliases=(
            "number base",
            "decimal",
            "denary",
            "base ten",
            "binary",
            "base two",
            "hexadecimal",
            "base sixteen",
        ),
        paper="Paper 2",
        source_pages=(18,),
    ),
    _concept(
        concept_id="aqa_3_3_1_binary_representation",
        official_reference="3.3.1",
        chapter_reference="3.3",
        chapter_title=AQA_CHAPTERS["3.3"],
        official_title="Number bases",
        label="Binary representation of data and instructions",
        description=(
            "Understand that computers use binary bit patterns to "
            "represent data and instructions."
        ),
        aliases=(
            "computers use binary",
            "binary representation",
            "bit pattern",
            "represent data in binary",
            "binary instructions",
        ),
        paper="Paper 2",
        source_pages=(18,),
        parent_concept_id="aqa_3_3_1_number_bases",
    ),
    _concept(
        concept_id="aqa_3_3_1_hexadecimal_use",
        official_reference="3.3.1",
        chapter_reference="3.3",
        chapter_title=AQA_CHAPTERS["3.3"],
        official_title="Number bases",
        label="Why hexadecimal is used",
        description=(
            "Explain why hexadecimal is useful in computer science."
        ),
        aliases=(
            "why hexadecimal is used",
            "hex is shorter",
            "compact binary representation",
            "easier to read than binary",
            "hexadecimal in computer science",
        ),
        paper="Paper 2",
        source_pages=(18,),
        parent_concept_id="aqa_3_3_1_number_bases",
    ),
    _concept(
        concept_id="aqa_3_3_2_base_conversion",
        official_reference="3.3.2",
        chapter_reference="3.3",
        chapter_title=AQA_CHAPTERS["3.3"],
        official_title="Converting between number bases",
        label="Converting between binary, decimal and hexadecimal",
        description=(
            "Convert whole-number values in both directions between binary, "
            "decimal and hexadecimal, within the specified range."
        ),
        aliases=(
            "binary to decimal",
            "binary to denary",
            "decimal to binary",
            "denary to binary",
            "binary to hexadecimal",
            "hexadecimal to binary",
            "decimal to hexadecimal",
            "hexadecimal to decimal",
            "convert number bases",
        ),
        paper="Paper 2",
        source_pages=(18,),
    ),
    _concept(
        concept_id="aqa_3_3_3_bits_bytes",
        official_reference="3.3.3",
        chapter_reference="3.3",
        chapter_title=AQA_CHAPTERS["3.3"],
        official_title="Units of information",
        label="Bits and bytes",
        description=(
            "Know that a bit is a binary digit and a byte is a group of "
            "eight bits."
        ),
        aliases=(
            "byte",
            "eight bits",
            "8 bits",
            "binary digit",
            "unit of information",
        ),
        paper="Paper 2",
        source_pages=(18,),
    ),
    _concept(
        concept_id="aqa_3_3_3_storage_units",
        official_reference="3.3.3",
        chapter_reference="3.3",
        chapter_title=AQA_CHAPTERS["3.3"],
        official_title="Units of information",
        label="Decimal storage units",
        description=(
            "Know and compare kilo, mega, giga and tera quantities using "
            "decimal powers of ten."
        ),
        aliases=(
            "kilobyte",
            "megabyte",
            "gigabyte",
            "terabyte",
            "kb",
            "mb",
            "gb",
            "tb",
            "storage units",
            "decimal prefixes",
        ),
        paper="Paper 2",
        source_pages=(19,),
    ),
    _concept(
        concept_id="aqa_3_3_4_binary_addition",
        official_reference="3.3.4",
        chapter_reference="3.3",
        chapter_title=AQA_CHAPTERS["3.3"],
        official_title="Binary arithmetic",
        label="Binary addition",
        description=(
            "Add together up to three binary numbers within the stated "
            "eight-bit limits."
        ),
        aliases=(
            "binary addition",
            "add binary numbers",
            "binary sum",
            "carry in binary",
            "add bits",
        ),
        paper="Paper 2",
        source_pages=(19,),
    ),
    _concept(
        concept_id="aqa_3_3_4_binary_shifts",
        official_reference="3.3.4",
        chapter_reference="3.3",
        chapter_title=AQA_CHAPTERS["3.3"],
        official_title="Binary arithmetic",
        label="Logical binary shifts",
        description=(
            "Apply logical binary shifts and explain their use for "
            "multiplication or division by powers of two."
        ),
        aliases=(
            "binary shift",
            "logical shift",
            "left shift",
            "right shift",
            "multiply by powers of two",
            "divide by powers of two",
        ),
        paper="Paper 2",
        source_pages=(19,),
    ),
    _concept(
        concept_id="aqa_3_3_5_character_encoding",
        official_reference="3.3.5",
        chapter_reference="3.3",
        chapter_title=AQA_CHAPTERS["3.3"],
        official_title="Character encoding",
        label="ASCII and Unicode character encoding",
        description=(
            "Understand character sets, 7-bit ASCII and Unicode, convert "
            "between characters and codes, and explain the advantages of "
            "Unicode."
        ),
        aliases=(
            "character encoding",
            "character set",
            "ascii",
            "7 bit ascii",
            "unicode",
            "character code",
            "code point",
            "unicode versus ascii",
        ),
        paper="Paper 2",
        source_pages=(19, 20),
    ),
    _concept(
        concept_id="aqa_3_3_6_bitmap_images",
        official_reference="3.3.6",
        chapter_reference="3.3",
        chapter_title=AQA_CHAPTERS["3.3"],
        official_title="Representing images",
        label="Bitmap images, pixels and colour depth",
        description=(
            "Explain how bitmap images use pixels, image dimensions and "
            "colour depth."
        ),
        aliases=(
            "bitmap image",
            "pixel",
            "picture element",
            "image size",
            "image resolution",
            "width by height",
            "colour depth",
            "color depth",
            "bits per pixel",
        ),
        paper="Paper 2",
        source_pages=(20,),
    ),
    _concept(
        concept_id="aqa_3_3_6_image_file_size",
        official_reference="3.3.6",
        chapter_reference="3.3",
        chapter_title=AQA_CHAPTERS["3.3"],
        official_title="Representing images",
        label="Bitmap file-size calculations",
        description=(
            "Calculate bitmap file size from width, height and colour "
            "depth, and explain how pixels and colour depth affect size."
        ),
        aliases=(
            "bitmap file size",
            "image file size",
            "width times height times colour depth",
            "w h d",
            "pixels affect file size",
            "colour depth affects file size",
        ),
        paper="Paper 2",
        source_pages=(20,),
        parent_concept_id="aqa_3_3_6_bitmap_images",
    ),
    _concept(
        concept_id="aqa_3_3_6_bitmap_binary_conversion",
        official_reference="3.3.6",
        chapter_reference="3.3",
        chapter_title=AQA_CHAPTERS["3.3"],
        official_title="Representing images",
        label="Converting between bitmap images and binary data",
        description=(
            "Convert simple binary patterns into bitmap images and simple "
            "bitmap images into binary data."
        ),
        aliases=(
            "binary bitmap",
            "bitmap to binary",
            "binary to bitmap",
            "draw pixels from binary",
            "convert image to binary",
        ),
        paper="Paper 2",
        source_pages=(21,),
        parent_concept_id="aqa_3_3_6_bitmap_images",
    ),
    _concept(
        concept_id="aqa_3_3_7_sound_sampling",
        official_reference="3.3.7",
        chapter_reference="3.3",
        chapter_title=AQA_CHAPTERS["3.3"],
        official_title="Representing sound",
        label="Digital sound sampling",
        description=(
            "Explain analogue-to-digital sound conversion using samples, "
            "sampling rate and sample resolution."
        ),
        aliases=(
            "digital sound",
            "analogue sound",
            "analog sound",
            "sound sampling",
            "sample rate",
            "sampling rate",
            "sample resolution",
            "sample depth",
            "amplitude sample",
        ),
        paper="Paper 2",
        source_pages=(21,),
    ),
    _concept(
        concept_id="aqa_3_3_7_sound_file_size",
        official_reference="3.3.7",
        chapter_reference="3.3",
        chapter_title=AQA_CHAPTERS["3.3"],
        official_title="Representing sound",
        label="Sound file-size calculations",
        description=(
            "Calculate sound file size from sampling rate, sample "
            "resolution and duration."
        ),
        aliases=(
            "sound file size",
            "sampling rate times resolution times seconds",
            "audio file size",
            "rate res secs",
            "calculate sound size",
        ),
        paper="Paper 2",
        source_pages=(21,),
        parent_concept_id="aqa_3_3_7_sound_sampling",
    ),
    _concept(
        concept_id="aqa_3_3_8_compression",
        official_reference="3.3.8",
        chapter_reference="3.3",
        chapter_title=AQA_CHAPTERS["3.3"],
        official_title="Data compression",
        label="Data compression",
        description=(
            "Explain data compression, why it is used and that different "
            "compression methods exist."
        ),
        aliases=(
            "data compression",
            "compress data",
            "reduce file size",
            "compressed file",
            "why compress data",
        ),
        paper="Paper 2",
        source_pages=(21,),
    ),
    _concept(
        concept_id="aqa_3_3_8_huffman",
        official_reference="3.3.8",
        chapter_reference="3.3",
        chapter_title=AQA_CHAPTERS["3.3"],
        official_title="Data compression",
        label="Huffman coding",
        description=(
            "Explain and interpret Huffman coding and Huffman trees, and "
            "calculate compressed and uncompressed bit totals."
        ),
        aliases=(
            "huffman coding",
            "huffman tree",
            "huffman code",
            "variable length code",
            "compressed bits",
            "ascii bits",
        ),
        paper="Paper 2",
        source_pages=(21, 22),
        parent_concept_id="aqa_3_3_8_compression",
    ),
    _concept(
        concept_id="aqa_3_3_8_rle",
        official_reference="3.3.8",
        chapter_reference="3.3",
        chapter_title=AQA_CHAPTERS["3.3"],
        official_title="Data compression",
        label="Run-length encoding",
        description=(
            "Explain run-length encoding and represent data using "
            "frequency/data pairs."
        ),
        aliases=(
            "run length encoding",
            "rle",
            "frequency data pairs",
            "repeat count and value",
            "compress repeated values",
        ),
        paper="Paper 2",
        source_pages=(22,),
        parent_concept_id="aqa_3_3_8_compression",
    ),

    # =========================================================================
    # 3.4 COMPUTER SYSTEMS — PAPER 2
    # =========================================================================
    _concept(
        concept_id="aqa_3_4_1_hardware_software",
        official_reference="3.4.1",
        chapter_reference="3.4",
        chapter_title=AQA_CHAPTERS["3.4"],
        official_title="Hardware and software",
        label="Hardware and software",
        description=(
            "Define hardware and software and understand the relationship "
            "between them."
        ),
        aliases=(
            "hardware",
            "software",
            "physical components",
            "computer programs",
            "hardware and software relationship",
        ),
        paper="Paper 2",
        source_pages=(22,),
    ),
    _concept(
        concept_id="aqa_3_4_2_truth_tables_logic_gates",
        official_reference="3.4.2",
        chapter_reference="3.4",
        chapter_title=AQA_CHAPTERS["3.4"],
        official_title="Boolean logic",
        label="Logic gates and truth tables",
        description=(
            "Construct and interpret truth tables for NOT, AND, OR and XOR "
            "gates and combinations of those gates."
        ),
        aliases=(
            "logic gate",
            "truth table",
            "and gate",
            "or gate",
            "not gate",
            "xor gate",
            "logic circuit truth table",
        ),
        paper="Paper 2",
        source_pages=(22,),
    ),
    _concept(
        concept_id="aqa_3_4_2_logic_circuits",
        official_reference="3.4.2",
        chapter_reference="3.4",
        chapter_title=AQA_CHAPTERS["3.4"],
        official_title="Boolean logic",
        label="Logic circuit diagrams",
        description=(
            "Create, modify and interpret simple logic circuit diagrams "
            "using NOT, AND, OR and XOR."
        ),
        aliases=(
            "logic circuit",
            "circuit diagram",
            "draw logic gates",
            "interpret the circuit",
            "modify logic circuit",
        ),
        paper="Paper 2",
        source_pages=(23,),
        parent_concept_id="aqa_3_4_2_truth_tables_logic_gates",
    ),
    _concept(
        concept_id="aqa_3_4_2_boolean_expressions",
        official_reference="3.4.2",
        chapter_reference="3.4",
        chapter_title=AQA_CHAPTERS["3.4"],
        official_title="Boolean logic",
        label="Boolean expressions and circuits",
        description=(
            "Create and interpret Boolean expressions and convert between "
            "simple expressions and logic circuits."
        ),
        aliases=(
            "boolean expression",
            "logic expression",
            "expression from circuit",
            "circuit from expression",
            "a and b or not c",
        ),
        paper="Paper 2",
        source_pages=(23,),
        parent_concept_id="aqa_3_4_2_truth_tables_logic_gates",
    ),
    _concept(
        concept_id="aqa_3_4_3_software_classification",
        official_reference="3.4.3",
        chapter_reference="3.4",
        chapter_title=AQA_CHAPTERS["3.4"],
        official_title="Software classification",
        label="System software and application software",
        description=(
            "Explain system software and application software and give "
            "examples of both."
        ),
        aliases=(
            "system software",
            "application software",
            "software classification",
            "end user software",
            "platform software",
        ),
        paper="Paper 2",
        source_pages=(23,),
    ),
    _concept(
        concept_id="aqa_3_4_3_operating_systems_utilities",
        official_reference="3.4.3",
        chapter_reference="3.4",
        chapter_title=AQA_CHAPTERS["3.4"],
        official_title="Software classification",
        label="Operating systems and utility programs",
        description=(
            "Explain why operating systems and utilities are needed and "
            "describe management of processor, memory, I/O devices, "
            "applications and security."
        ),
        aliases=(
            "operating system",
            "os",
            "utility program",
            "processor management",
            "memory management",
            "device management",
            "application management",
            "operating system security",
        ),
        paper="Paper 2",
        source_pages=(24,),
        parent_concept_id="aqa_3_4_3_software_classification",
    ),
    _concept(
        concept_id="aqa_3_4_4_language_levels",
        official_reference="3.4.4",
        chapter_reference="3.4",
        chapter_title=AQA_CHAPTERS["3.4"],
        official_title=(
            "Classification of programming languages and translators"
        ),
        label="High-level and low-level programming languages",
        description=(
            "Compare high-level and low-level languages and explain "
            "machine code and assembly language."
        ),
        aliases=(
            "high level language",
            "low level language",
            "machine code",
            "assembly language",
            "one to one with machine code",
            "processor instruction set",
        ),
        paper="Paper 2",
        source_pages=(24,),
    ),
    _concept(
        concept_id="aqa_3_4_4_translators",
        official_reference="3.4.4",
        chapter_reference="3.4",
        chapter_title=AQA_CHAPTERS["3.4"],
        official_title=(
            "Classification of programming languages and translators"
        ),
        label="Compilers, interpreters and assemblers",
        description=(
            "Explain the differences between compiler, interpreter and "
            "assembler translators and when each is appropriate."
        ),
        aliases=(
            "compiler",
            "interpreter",
            "assembler",
            "program translator",
            "translate to machine code",
            "compiled code",
            "interpreted code",
        ),
        paper="Paper 2",
        source_pages=(25,),
    ),
    _concept(
        concept_id="aqa_3_4_5_cpu_von_neumann",
        official_reference="3.4.5",
        chapter_reference="3.4",
        chapter_title=AQA_CHAPTERS["3.4"],
        official_title="Systems architecture",
        label="CPU components and Von Neumann architecture",
        description=(
            "Explain main memory and CPU components including the ALU, "
            "control unit, clock, registers and buses."
        ),
        aliases=(
            "von neumann architecture",
            "cpu",
            "central processing unit",
            "arithmetic logic unit",
            "alu",
            "control unit",
            "clock",
            "register",
            "bus",
            "main memory",
        ),
        paper="Paper 2",
        source_pages=(25,),
    ),
    _concept(
        concept_id="aqa_3_4_5_cpu_performance",
        official_reference="3.4.5",
        chapter_reference="3.4",
        chapter_title=AQA_CHAPTERS["3.4"],
        official_title="Systems architecture",
        label="CPU performance",
        description=(
            "Explain how clock speed, number of processor cores and cache "
            "size affect CPU performance."
        ),
        aliases=(
            "cpu performance",
            "clock speed",
            "processor cores",
            "number of cores",
            "cache size",
            "faster processor",
        ),
        paper="Paper 2",
        source_pages=(25,),
        parent_concept_id="aqa_3_4_5_cpu_von_neumann",
    ),
    _concept(
        concept_id="aqa_3_4_5_fetch_execute_cycle",
        official_reference="3.4.5",
        chapter_reference="3.4",
        chapter_title=AQA_CHAPTERS["3.4"],
        official_title="Systems architecture",
        label="Fetch-decode-execute cycle",
        description=(
            "Explain how the CPU fetches, decodes and executes instructions "
            "stored in main memory."
        ),
        aliases=(
            "fetch execute cycle",
            "fetch decode execute",
            "instruction cycle",
            "fetch instruction",
            "decode instruction",
            "execute instruction",
        ),
        paper="Paper 2",
        source_pages=(25,),
        parent_concept_id="aqa_3_4_5_cpu_von_neumann",
    ),
    _concept(
        concept_id="aqa_3_4_5_memory",
        official_reference="3.4.5",
        chapter_reference="3.4",
        chapter_title=AQA_CHAPTERS["3.4"],
        official_title="Systems architecture",
        label="RAM, ROM, cache and registers",
        description=(
            "Explain the uses of RAM, ROM, cache and registers and compare "
            "main memory, secondary storage, volatile and non-volatile "
            "memory."
        ),
        aliases=(
            "ram",
            "rom",
            "cache memory",
            "register memory",
            "volatile memory",
            "non volatile memory",
            "main memory",
            "secondary storage",
            "ram versus rom",
        ),
        paper="Paper 2",
        source_pages=(26,),
    ),
    _concept(
        concept_id="aqa_3_4_5_secondary_storage",
        official_reference="3.4.5",
        chapter_reference="3.4",
        chapter_title=AQA_CHAPTERS["3.4"],
        official_title="Systems architecture",
        label="Secondary storage",
        description=(
            "Explain why secondary storage is required and compare solid "
            "state, optical and magnetic storage."
        ),
        aliases=(
            "secondary storage",
            "solid state storage",
            "ssd",
            "optical storage",
            "magnetic storage",
            "hard disk",
            "storage advantages disadvantages",
        ),
        paper="Paper 2",
        source_pages=(26,),
    ),
    _concept(
        concept_id="aqa_3_4_5_cloud_storage",
        official_reference="3.4.5",
        chapter_reference="3.4",
        chapter_title=AQA_CHAPTERS["3.4"],
        official_title="Systems architecture",
        label="Cloud storage",
        description=(
            "Explain cloud storage and compare remote cloud storage with "
            "local storage."
        ),
        aliases=(
            "cloud storage",
            "remote storage",
            "local storage",
            "store data online",
            "cloud versus local",
        ),
        paper="Paper 2",
        source_pages=(26,),
    ),
    _concept(
        concept_id="aqa_3_4_5_embedded_systems",
        official_reference="3.4.5",
        chapter_reference="3.4",
        chapter_title=AQA_CHAPTERS["3.4"],
        official_title="Systems architecture",
        label="Embedded systems",
        description=(
            "Explain how embedded systems differ from non-embedded systems "
            "and give examples."
        ),
        aliases=(
            "embedded system",
            "embedded computer",
            "dedicated system",
            "non embedded system",
            "computer inside a device",
        ),
        paper="Paper 2",
        source_pages=(26,),
    ),

    # =========================================================================
    # 3.5 FUNDAMENTALS OF COMPUTER NETWORKS — PAPER 2
    # =========================================================================
    _concept(
        concept_id="aqa_3_5_network_fundamentals",
        official_reference="3.5",
        chapter_reference="3.5",
        chapter_title=AQA_CHAPTERS["3.5"],
        official_title="Fundamentals of computer networks",
        label="Computer networks",
        description=(
            "Define computer networks and discuss their advantages and "
            "disadvantages."
        ),
        aliases=(
            "computer network",
            "network definition",
            "connected computers",
            "share resources",
            "advantages of networks",
            "disadvantages of networks",
        ),
        paper="Paper 2",
        source_pages=(27,),
    ),
    _concept(
        concept_id="aqa_3_5_network_types",
        official_reference="3.5",
        chapter_reference="3.5",
        chapter_title=AQA_CHAPTERS["3.5"],
        official_title="Fundamentals of computer networks",
        label="PAN, LAN and WAN",
        description=(
            "Describe personal, local and wide area networks, including "
            "typical scale, ownership and examples."
        ),
        aliases=(
            "pan",
            "personal area network",
            "bluetooth network",
            "lan",
            "local area network",
            "wan",
            "wide area network",
            "internet is a wan",
            "network types",
        ),
        paper="Paper 2",
        source_pages=(27,),
        parent_concept_id="aqa_3_5_network_fundamentals",
    ),
    _concept(
        concept_id="aqa_3_5_wired_wireless",
        official_reference="3.5",
        chapter_reference="3.5",
        chapter_title=AQA_CHAPTERS["3.5"],
        official_title="Fundamentals of computer networks",
        label="Wired and wireless networks",
        description=(
            "Compare wired and wireless networks, including copper and "
            "fibre cabling and appropriate uses."
        ),
        aliases=(
            "wired network",
            "wireless network",
            "wifi",
            "wi fi",
            "copper cable",
            "fibre cable",
            "fiber cable",
            "wired versus wireless",
        ),
        paper="Paper 2",
        source_pages=(27,),
        parent_concept_id="aqa_3_5_network_fundamentals",
    ),
    _concept(
        concept_id="aqa_3_5_topologies",
        official_reference="3.5",
        chapter_reference="3.5",
        chapter_title=AQA_CHAPTERS["3.5"],
        official_title="Fundamentals of computer networks",
        label="Star and bus network topologies",
        description=(
            "Describe, draw, compare and select star and bus LAN "
            "topologies."
        ),
        aliases=(
            "network topology",
            "star topology",
            "bus topology",
            "lan topology",
            "topology diagram",
            "central switch",
            "backbone cable",
        ),
        paper="Paper 2",
        source_pages=(27,),
        parent_concept_id="aqa_3_5_network_fundamentals",
    ),
    _concept(
        concept_id="aqa_3_5_protocols",
        official_reference="3.5",
        chapter_reference="3.5",
        chapter_title=AQA_CHAPTERS["3.5"],
        official_title="Fundamentals of computer networks",
        label="Network protocols",
        description=(
            "Define protocols and explain the purpose and use of Ethernet, "
            "Wi-Fi, TCP, UDP, IP, HTTP, HTTPS, FTP, SMTP and IMAP."
        ),
        aliases=(
            "network protocol",
            "protocol",
            "ethernet",
            "wifi protocol",
            "tcp",
            "udp",
            "ip protocol",
            "http",
            "https",
            "ftp",
            "smtp",
            "imap",
            "email protocol",
        ),
        paper="Paper 2",
        source_pages=(27, 28),
        parent_concept_id="aqa_3_5_network_fundamentals",
    ),
    _concept(
        concept_id="aqa_3_5_network_security",
        official_reference="3.5",
        chapter_reference="3.5",
        chapter_title=AQA_CHAPTERS["3.5"],
        official_title="Fundamentals of computer networks",
        label="Network security methods",
        description=(
            "Explain network security using authentication, encryption, "
            "firewalls and MAC address filtering, including how controls "
            "work together."
        ),
        aliases=(
            "network security",
            "authentication",
            "encryption",
            "firewall",
            "mac address filtering",
            "block network traffic",
            "allow network traffic",
            "security layers",
        ),
        paper="Paper 2",
        source_pages=(28,),
        parent_concept_id="aqa_3_5_network_fundamentals",
    ),
    _concept(
        concept_id="aqa_3_5_tcp_ip_model",
        official_reference="3.5",
        chapter_reference="3.5",
        chapter_title=AQA_CHAPTERS["3.5"],
        official_title="Fundamentals of computer networks",
        label="Four-layer TCP/IP model",
        description=(
            "Describe the application, transport, internet and link layers "
            "and place common protocols at the correct layer."
        ),
        aliases=(
            "tcp ip model",
            "four layer model",
            "4 layer model",
            "application layer",
            "transport layer",
            "internet layer",
            "link layer",
            "network access layer",
        ),
        paper="Paper 2",
        source_pages=(29,),
        parent_concept_id="aqa_3_5_network_fundamentals",
    ),

    # =========================================================================
    # 3.6 CYBER SECURITY — PAPER 2
    # =========================================================================
    _concept(
        concept_id="aqa_3_6_1_cyber_security",
        official_reference="3.6.1",
        chapter_reference="3.6",
        chapter_title=AQA_CHAPTERS["3.6"],
        official_title="Fundamentals of cyber security",
        label="Cyber security fundamentals",
        description=(
            "Define cyber security and describe its purpose in protecting "
            "networks, computers, programs and data."
        ),
        aliases=(
            "cyber security",
            "cybersecurity",
            "protect networks",
            "protect computers",
            "protect data",
            "unauthorised access",
            "unauthorized access",
        ),
        paper="Paper 2",
        source_pages=(29,),
    ),
    _concept(
        concept_id="aqa_3_6_2_cyber_threats",
        official_reference="3.6.2",
        chapter_reference="3.6",
        chapter_title=AQA_CHAPTERS["3.6"],
        official_title="Cyber security threats",
        label="Cyber security threats",
        description=(
            "Explain social engineering, malware, pharming, weak or "
            "default passwords, misconfigured access rights, removable "
            "media and unpatched software as threats."
        ),
        aliases=(
            "cyber threat",
            "security threat",
            "pharming",
            "weak password",
            "default password",
            "misconfigured access rights",
            "removable media",
            "unpatched software",
            "outdated software",
        ),
        paper="Paper 2",
        source_pages=(30,),
        parent_concept_id="aqa_3_6_1_cyber_security",
    ),
    _concept(
        concept_id="aqa_3_6_2_penetration_testing",
        official_reference="3.6.2",
        chapter_reference="3.6",
        chapter_title=AQA_CHAPTERS["3.6"],
        official_title="Cyber security threats",
        label="Penetration testing",
        description=(
            "Explain penetration testing and distinguish testing that "
            "simulates an internal attack from testing that simulates an "
            "external attack."
        ),
        aliases=(
            "penetration testing",
            "pen test",
            "ethical hacking",
            "test system security",
            "internal attack",
            "external attack",
            "test without credentials",
        ),
        paper="Paper 2",
        source_pages=(30,),
        parent_concept_id="aqa_3_6_1_cyber_security",
    ),
    _concept(
        concept_id="aqa_3_6_2_1_social_engineering",
        official_reference="3.6.2.1",
        chapter_reference="3.6",
        chapter_title=AQA_CHAPTERS["3.6"],
        official_title="Social engineering",
        label="Social engineering, blagging, phishing and shouldering",
        description=(
            "Define social engineering, explain how it is prevented and "
            "describe blagging, phishing and shouldering."
        ),
        aliases=(
            "social engineering",
            "blagging",
            "pretexting",
            "phishing",
            "fake email",
            "fraudulent message",
            "shouldering",
            "shoulder surfing",
            "steal confidential information",
        ),
        paper="Paper 2",
        source_pages=(30,),
        parent_concept_id="aqa_3_6_2_cyber_threats",
    ),
    _concept(
        concept_id="aqa_3_6_2_2_malware",
        official_reference="3.6.2.2",
        chapter_reference="3.6",
        chapter_title=AQA_CHAPTERS["3.6"],
        official_title="Malicious code (malware)",
        label="Malware, viruses, Trojans and spyware",
        description=(
            "Define malware, explain protection against it and describe "
            "computer viruses, Trojans and spyware."
        ),
        aliases=(
            "malware",
            "malicious code",
            "computer virus",
            "virus",
            "trojan",
            "spyware",
            "hostile software",
            "intrusive software",
        ),
        paper="Paper 2",
        source_pages=(31,),
        parent_concept_id="aqa_3_6_2_cyber_threats",
    ),
    _concept(
        concept_id="aqa_3_6_3_security_measures",
        official_reference="3.6.3",
        chapter_reference="3.6",
        chapter_title=AQA_CHAPTERS["3.6"],
        official_title=(
            "Methods to detect and prevent cyber security threats"
        ),
        label="Cyber security prevention measures",
        description=(
            "Explain biometric measures, password systems, CAPTCHA, email "
            "identity confirmations and automatic software updates."
        ),
        aliases=(
            "biometric security",
            "fingerprint security",
            "password system",
            "captcha",
            "email confirmation",
            "confirm user identity",
            "automatic software update",
            "security update",
        ),
        paper="Paper 2",
        source_pages=(31,),
        parent_concept_id="aqa_3_6_1_cyber_security",
    ),

    # =========================================================================
    # 3.7 RELATIONAL DATABASES AND SQL — PAPER 2
    # =========================================================================
    _concept(
        concept_id="aqa_3_7_1_database_fundamentals",
        official_reference="3.7.1",
        chapter_reference="3.7",
        chapter_title=AQA_CHAPTERS["3.7"],
        official_title="Relational databases",
        label="Database and relational database concepts",
        description=(
            "Explain databases and relational databases."
        ),
        aliases=(
            "database",
            "relational database",
            "database concept",
            "related tables",
            "store structured data",
        ),
        paper="Paper 2",
        source_pages=(31,),
    ),
    _concept(
        concept_id="aqa_3_7_1_database_structure",
        official_reference="3.7.1",
        chapter_reference="3.7",
        chapter_title=AQA_CHAPTERS["3.7"],
        official_title="Relational databases",
        label="Tables, records, fields and data types",
        description=(
            "Understand tables, records, fields and data types in a "
            "relational database."
        ),
        aliases=(
            "database table",
            "record",
            "database field",
            "column",
            "row",
            "database data type",
        ),
        paper="Paper 2",
        source_pages=(32,),
        parent_concept_id="aqa_3_7_1_database_fundamentals",
    ),
    _concept(
        concept_id="aqa_3_7_1_keys",
        official_reference="3.7.1",
        chapter_reference="3.7",
        chapter_title=AQA_CHAPTERS["3.7"],
        official_title="Relational databases",
        label="Primary keys and foreign keys",
        description=(
            "Understand primary keys and foreign keys in relational "
            "databases."
        ),
        aliases=(
            "primary key",
            "foreign key",
            "unique identifier",
            "link tables",
            "related table key",
        ),
        paper="Paper 2",
        source_pages=(32,),
        parent_concept_id="aqa_3_7_1_database_fundamentals",
    ),
    _concept(
        concept_id="aqa_3_7_1_redundancy_inconsistency",
        official_reference="3.7.1",
        chapter_reference="3.7",
        chapter_title=AQA_CHAPTERS["3.7"],
        official_title="Relational databases",
        label="Data redundancy and inconsistency",
        description=(
            "Explain how relational databases help reduce data "
            "inconsistency and data redundancy."
        ),
        aliases=(
            "data redundancy",
            "data inconsistency",
            "duplicate data",
            "remove repeated data",
            "relational database benefits",
        ),
        paper="Paper 2",
        source_pages=(32,),
        parent_concept_id="aqa_3_7_1_database_fundamentals",
    ),
    _concept(
        concept_id="aqa_3_7_2_sql_select",
        official_reference="3.7.2",
        chapter_reference="3.7",
        chapter_title=AQA_CHAPTERS["3.7"],
        official_title="Structured query language (SQL)",
        label="SQL data retrieval",
        description=(
            "Retrieve relational database data using SELECT, FROM, WHERE "
            "and ORDER BY with ascending or descending order."
        ),
        aliases=(
            "sql select",
            "select from",
            "where clause",
            "order by",
            "ascending order",
            "descending order",
            "retrieve database data",
            "sql query",
        ),
        paper="Paper 2",
        source_pages=(32,),
    ),
    _concept(
        concept_id="aqa_3_7_2_sql_insert",
        official_reference="3.7.2",
        chapter_reference="3.7",
        chapter_title=AQA_CHAPTERS["3.7"],
        official_title="Structured query language (SQL)",
        label="SQL INSERT",
        description=(
            "Insert data into a relational database using INSERT INTO and "
            "VALUES."
        ),
        aliases=(
            "insert into",
            "sql insert",
            "values clause",
            "add database record",
            "insert a row",
        ),
        paper="Paper 2",
        source_pages=(32,),
    ),
    _concept(
        concept_id="aqa_3_7_2_sql_update_delete",
        official_reference="3.7.2",
        chapter_reference="3.7",
        chapter_title=AQA_CHAPTERS["3.7"],
        official_title="Structured query language (SQL)",
        label="SQL UPDATE and DELETE",
        description=(
            "Edit and delete relational database data using UPDATE, SET, "
            "DELETE FROM and WHERE."
        ),
        aliases=(
            "sql update",
            "update set",
            "delete from",
            "sql delete",
            "edit database data",
            "delete a record",
            "where condition",
        ),
        paper="Paper 2",
        source_pages=(32,),
    ),

    # =========================================================================
    # 3.8 ETHICAL, LEGAL AND ENVIRONMENTAL IMPACTS — PAPER 2
    # =========================================================================
    _concept(
        concept_id="aqa_3_8_impacts",
        official_reference="3.8",
        chapter_reference="3.8",
        chapter_title=AQA_CHAPTERS["3.8"],
        official_title=AQA_CHAPTERS["3.8"],
        label="Ethical, legal and environmental impacts of technology",
        description=(
            "Explain current ethical, legal and environmental impacts and "
            "risks of digital technology on society."
        ),
        aliases=(
            "ethical impact",
            "legal impact",
            "environmental impact",
            "digital technology on society",
            "social impact of technology",
            "technology risks",
        ),
        paper="Paper 2",
        source_pages=(33,),
    ),
    _concept(
        concept_id="aqa_3_8_privacy",
        official_reference="3.8",
        chapter_reference="3.8",
        chapter_title=AQA_CHAPTERS["3.8"],
        official_title=AQA_CHAPTERS["3.8"],
        label="Privacy and access to personal data",
        description=(
            "Consider privacy issues and competing arguments about access "
            "to private data by governments and security services."
        ),
        aliases=(
            "data privacy",
            "personal privacy",
            "private data",
            "government access to data",
            "security services",
            "surveillance",
            "citizen privacy",
        ),
        paper="Paper 2",
        source_pages=(33,),
        parent_concept_id="aqa_3_8_impacts",
    ),
    _concept(
        concept_id="aqa_3_8_contexts",
        official_reference="3.8",
        chapter_reference="3.8",
        chapter_title=AQA_CHAPTERS["3.8"],
        official_title=AQA_CHAPTERS["3.8"],
        label="Societal impact contexts",
        description=(
            "Apply ethical, legal, environmental and privacy principles to "
            "cyber security, mobile and wireless technologies, cloud "
            "storage, hacking, wearables, implants and autonomous vehicles."
        ),
        aliases=(
            "mobile technology impact",
            "wireless networking impact",
            "cloud storage impact",
            "hacking impact",
            "wearable technology",
            "computer implant",
            "autonomous vehicle",
            "self driving car",
        ),
        paper="Paper 2",
        source_pages=(33,),
        parent_concept_id="aqa_3_8_impacts",
    ),
)


# =============================================================================
# LOOKUP INDEXES
# =============================================================================

CONCEPT_BY_ID: dict[str, CSConcept] = {
    concept.concept_id: concept
    for concept in CS_CONCEPTS
}


CONCEPTS_BY_REFERENCE: dict[str, tuple[CSConcept, ...]] = {
    reference: tuple(concepts)
    for reference, concepts in (
        lambda grouped: grouped.items()
    )(
        (
            lambda grouped: [
                grouped[concept.official_reference].append(concept)
                for concept in CS_CONCEPTS
            ]
            and grouped
        )(
            defaultdict(list)
        )
    )
}


CONCEPTS_BY_CHAPTER: dict[str, tuple[CSConcept, ...]] = {
    chapter_reference: tuple(concepts)
    for chapter_reference, concepts in (
        lambda grouped: grouped.items()
    )(
        (
            lambda grouped: [
                grouped[concept.chapter_reference].append(concept)
                for concept in CS_CONCEPTS
            ]
            and grouped
        )(
            defaultdict(list)
        )
    )
}


def get_concept(concept_id: str) -> CSConcept:
    """
    Return one catalogue item by its stable internal concept ID.
    """

    try:
        return CONCEPT_BY_ID[concept_id]

    except KeyError as exc:
        raise KeyError(
            f"Unknown AQA concept_id: {concept_id}"
        ) from exc


def get_concepts_by_reference(
    official_reference: str,
) -> tuple[CSConcept, ...]:
    """
    Return all searchable concepts linked to one official AQA section.
    """

    return CONCEPTS_BY_REFERENCE.get(
        official_reference,
        (),
    )


def get_concepts_by_chapter(
    chapter_reference: str,
) -> tuple[CSConcept, ...]:
    """
    Return all searchable concepts inside an official AQA chapter.
    """

    return CONCEPTS_BY_CHAPTER.get(
        chapter_reference,
        (),
    )


def validate_catalog() -> None:
    """
    Fail early if the catalogue contains structural errors.
    """

    concept_ids: set[str] = set()

    for concept in CS_CONCEPTS:
        if concept.concept_id in concept_ids:
            raise ValueError(
                f"Duplicate concept_id: {concept.concept_id}"
            )

        concept_ids.add(
            concept.concept_id
        )

        if concept.chapter_reference not in AQA_CHAPTERS:
            raise ValueError(
                "Unknown chapter reference "
                f"{concept.chapter_reference} "
                f"for {concept.concept_id}"
            )

        if (
            concept.chapter_title
            != AQA_CHAPTERS[
                concept.chapter_reference
            ]
        ):
            raise ValueError(
                "Chapter title mismatch for "
                f"{concept.concept_id}"
            )

        if not concept.official_reference.startswith(
            concept.chapter_reference
        ):
            raise ValueError(
                "Official reference does not belong to chapter for "
                f"{concept.concept_id}"
            )

        if not concept.aliases:
            raise ValueError(
                f"No aliases defined for {concept.concept_id}"
            )

        for pattern in concept.match_patterns:
            if not pattern.label.strip():
                raise ValueError(
                    f"Empty pattern label for {concept.concept_id}"
                )

            if not 0.0 <= pattern.weight <= 1.0:
                raise ValueError(
                    f"Invalid pattern weight for {concept.concept_id}"
                )

            try:
                re.compile(pattern.regex, re.IGNORECASE)
            except re.error as exc:
                raise ValueError(
                    f"Invalid regex pattern for {concept.concept_id}: "
                    f"{pattern.regex}"
                ) from exc

        if concept.parent_concept_id is not None:
            if concept.parent_concept_id == concept.concept_id:
                raise ValueError(
                    f"Concept cannot parent itself: {concept.concept_id}"
                )

    missing_parents = {
        concept.parent_concept_id
        for concept in CS_CONCEPTS
        if (
            concept.parent_concept_id is not None
            and concept.parent_concept_id not in concept_ids
        )
    }

    if missing_parents:
        raise ValueError(
            "Missing parent concepts: "
            + ", ".join(
                sorted(missing_parents)
            )
        )

    missing_chapters = (
        set(AQA_CHAPTERS)
        - set(CONCEPTS_BY_CHAPTER)
    )

    if missing_chapters:
        raise ValueError(
            "No catalogue concepts for chapters: "
            + ", ".join(
                sorted(missing_chapters)
            )
        )


validate_catalog()