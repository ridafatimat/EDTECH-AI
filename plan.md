# EDTech AI Multi-Agent System — Project Plan

## 1. Project Overview

This project is an AI-powered EDTech system for analysing lesson transcripts, identifying the topics taught, mapping those topics to the official AQA GCSE Computer Science specification, and retrieving relevant past-paper questions and mark schemes. If suitable assessment material is unavailable, the system may generate a syllabus-grounded quiz.

The system is divided into two agents:

- **Agent 1 — Transcript and Syllabus Mapping**
- **Agent 2 — Assessment Retrieval and Quiz Generation**

The main design principle is to use deterministic rules and embeddings for the normal pipeline, while keeping the LLM as a fallback for uncertain cases.

---

## 2. High-Level Workflow

```text
Raw lesson transcript
        ↓
Module 1 — Lightweight preprocessing
        ↓
Cleaned transcript
        ↓
Module 2 — Guarded semantic chunking
        ↓
Meaningful transcript chunks
        ↓
Module 3 — Rough topic extraction
        ↓
Computer Science relevance filtering
        ↓
Global topic merging and deduplication
        ↓
Module 4 — Chapter identification
        ↓
Module 5 — Topic and subtopic identification
        ↓
Module 6 — Official AQA syllabus mapping
        ↓
Structured Agent 1 output
        ↓
Agent 2 — Past-paper and mark-scheme retrieval
        ↓
Assessment pack or generated quiz
```

---

# Agent 1 — Transcript and Syllabus Mapping

## 3. Agent 1 Objective

Agent 1 receives a raw teacher-student lesson transcript and identifies what was taught. It does not immediately force the full transcript into one syllabus topic. It first cleans and chunks the transcript, extracts rough topic candidates from each chunk, filters non-Computer-Science content, merges repeated topics, and then maps the retained topics to the official AQA specification.

### Planned Agent 1 Output

```json
{
  "qualification": "GCSE",
  "exam_board": "AQA",
  "subject": "Computer Science",
  "lesson_topics": [
    {
      "rough_topic": "tracing a while loop over an array",
      "syllabus_chapter": "Programming",
      "syllabus_topic": "Programming concepts",
      "syllabus_subtopic": "Iteration and arrays",
      "confidence": 0.91,
      "source_chunks": [3, 4, 5]
    }
  ]
}
```

---

## 4. Module 1 — Lightweight Transcript Preprocessing

### Objective

Clean obvious transcript noise without rewriting, summarising, correcting, or interpreting the lesson content.

### Current Processing Steps

1. Validate that the transcript is not empty.
2. Remove timestamp formats such as:
   - `00:12`
   - `01:05:31`
   - `[00:12]`
   - timestamp ranges
   - `Teacher - 16:00:01:`
3. Remove speaker labels such as:
   - Teacher
   - Student
   - Instructor
   - Tutor
   - Lecturer
   - Speaker 1
4. Remove known non-verbal artefacts such as:
   - background noise
   - keyboard or chair noise
   - audio glitches
   - notification sound/noise
   - screen-sharing events
   - connection loss/restoration
   - music, laughter, silence, and crosstalk
5. Remove uncertainty markers without guessing missing speech:
   - `[unclear]`
   - `[unknown]`
   - `???`
   - `<unk>`
6. Remove conservative fillers:
   - `um`
   - `uh`
   - `erm`
   - `hmm`
7. Preserve meaningful affirmation tokens:
   - `uh-huh`
   - `um-hmm`
8. Compress three or more immediately repeated words.
9. Remove consecutive duplicate sentences:
   - within the same paragraph
   - across adjacent lines or DOCX paragraphs
10. Normalise whitespace and punctuation spacing.
11. Return detailed cleaning statistics and warnings.

### Why the Initial Hybrid Approach Was Changed

The initial plan included hybrid preprocessing with optional LLM refinement. Testing on a real lesson transcript showed that most problems were structural rather than semantic. The transcript mainly contained fillers, ASR repetition, noise, speaker/timestamp variation, and connection artefacts.

Using an LLM on every run would add:

- unnecessary cost
- higher latency
- unpredictable rewriting
- risk of changing programming terminology
- risk of deleting context
- risk of guessing unclear speech

The final decision is:

```text
Routine preprocessing → deterministic rules
Uncertain educational meaning → handled later
LLM → fallback only when genuinely needed
```

### Current Status

- **Status:** Finalised and locked
- **Current rating:** approximately **9.5/10**
- **Processing speed:** generally a few milliseconds per transcript
- **Regression tests:** passed

---

## 5. Module 2 — Guarded Semantic Chunking

### Objective

Divide the cleaned transcript into coherent teaching sections while preserving enough context for later topic extraction.

Module 2 does not assume:

```text
1 chunk = 1 exact syllabus topic
```

A coherent chunk may contain multiple related topics. The next module will extract one or more topic candidates from each chunk.

### Current Chunking Process

1. Split the cleaned transcript into sentences.
2. Group nearby sentences into temporary semantic units.
3. Generate embeddings for each semantic unit.
4. Compare neighbouring units using cosine similarity.
5. Detect possible semantic changes.
6. Combine similarity evidence with transition phrases.
7. Apply minimum, target, and maximum chunk-size guardrails.
8. Add limited overlap only when maximum size forces a split.
9. Return structured chunks with boundary metadata.

### Final Semantic Unit Size

```text
Approximately 60 words
```

This prevents very short responses such as “okay,” “yes,” or “right” from being embedded individually.

### Final Chunk Size Configuration

```text
Minimum chunk size: 150 words
Target chunk size:  325 words
Maximum chunk size: 550 words
```

### Strong Transition Minimum

```text
80 words
```

A strong transition such as:

```text
“Let’s move on to the next question.”
```

can justify a smaller chunk after at least 80 words.

### Selective Overlap

Overlap is added only after a forced maximum-size split.

```text
Maximum overlap: approximately 45 words
Maximum overlap: 2 complete sentences
```

Semantic topic changes receive no overlap because a genuine boundary should remain clean.

---

## 6. Adaptive Semantic Threshold

A fixed similarity threshold was initially used, but testing showed that different transcripts and embedding models produce different cosine-similarity distributions.

### Initial Thresholds

```text
First version: 0.45
Later version: 0.38
```

These caused over-chunking in long conversational explanations.

### Final Adaptive Approach

1. Calculate cosine similarity between each pair of neighbouring semantic units.
2. Examine the similarity distribution of the current transcript.
3. Use the lowest 15% region as likely semantic changes.
4. Clamp the final threshold within a safe range.

```text
Boundary percentile: 15%
Minimum threshold:   0.10
Maximum threshold:   0.45
```

This allows every transcript to use a threshold suited to its own language style and noise level.

### Soft Transition Logic

```text
Soft transition margin: 0.10
Soft similarity ceiling: 0.35
```

If the adaptive threshold is `0.20`, a soft transition may be accepted up to approximately `0.30`, but never above `0.35`.

### Strong Transition Examples

- next chapter
- next topic
- next question
- start chapter
- move on to the next question
- let’s move on to the next thing

### Soft Transition Examples

- let’s discuss
- let’s look at
- moving on
- let’s talk about

The phrase:

```text
“Before we move on...”
```

was removed from the transition rules because it normally means the current topic is still continuing.

---

## 7. Embedding Model Decision

### Initial Model

```text
Qwen/Qwen3-Embedding-0.6B
```

It produced good semantic results but was too slow on the current CPU setup.

### Qwen Timing

```text
Warm run: approximately 157 seconds
```

### Final Model for Module 2

```text
sentence-transformers/all-MiniLM-L6-v2
```

### MiniLM Timing

```text
First transcript in process: several seconds including model initialisation
Subsequent transcripts: approximately 0.2–0.8 seconds
```

### Final Model Strategy

```text
Module 2 semantic chunking
→ all-MiniLM-L6-v2

Later syllabus retrieval and mapping
→ Qwen/Qwen3-Embedding-0.6B
```

### Current Status

- **Status:** Finalised and locked
- **Current rating:** approximately **9/10**
- **MiniLM speed rating:** **10/10**
- **Regression tests:** passed

---

## 8. Module 1 and Module 2 Testing

The pipeline was tested on five transcripts:

1. Clean Data Representation transcript
2. Clean Networks and Cybersecurity transcript
3. Algorithms and Programming stress transcript
4. Raw noisy Networks and Cybersecurity transcript
5. Raw noisy Algorithms and Programming transcript

### Test Coverage

The transcripts included:

- multiple timestamp styles
- multiple speaker-label formats
- fillers
- repeated words
- repeated sentences
- duplicate sentences across paragraphs
- ASR artefacts
- connection problems
- long same-topic explanations
- real topic transitions
- misleading transition phrases
- casual conversation
- multiple related Computer Science topics
- deliberately noisy transcript structure

### Final Regression Tests

- duplicate sentences across lines are removed
- `uh-huh` and `um-hmm` are preserved
- notification noise is removed
- “before we move on” is not treated as a boundary
- genuine next-question transitions are still detected

---

# Next Stage — Rough Topic Extraction

## 9. Module 3 — Rough Topic Candidate Extraction

### Objective

Extract one or more rough lesson topics from every chunk before forcing the content into the official syllabus.

### Example

```text
Chunk content:
Teacher explains a while loop traversing an array,
tracks index K, and discusses statement execution.

Possible rough topics:
- array traversal
- while loops
- code tracing
- loop conditions
```

### Planned Output Per Chunk

```json
{
  "chunk_id": 4,
  "topic_candidates": [
    {
      "topic": "array traversal using a while loop",
      "confidence": 0.92
    },
    {
      "topic": "code tracing and statement execution",
      "confidence": 0.86
    }
  ]
}
```

### Planned Extraction Strategy

Primary route:

1. keyword and phrase extraction
2. embedding similarity with a Computer Science concept vocabulary
3. technical-term detection
4. confidence scoring

Fallback route:

```text
Low-confidence or ambiguous chunk
        ↓
GPT-OSS-120B refinement
```

The LLM will not be called for every chunk.

---

## 10. Computer Science Relevance Filtering

After rough topics are extracted:

```text
Topic candidate
      ↓
Computer Science related?
   ├── No  → discard
   └── Yes → retain
```

### Example

A mixed chunk may produce:

```text
holiday
weather
arrays
ArrayLists
```

The filter should return:

```text
holiday      → reject
weather      → reject
arrays       → retain
ArrayLists   → retain
```

This is safer than classifying the entire chunk as either CS or non-CS.

---

## 11. Global Topic Merge and Deduplication

The same topic may continue across several chunks.

Example:

```text
Chunk 3 → arrays, while loops
Chunk 4 → array traversal, code tracing
Chunk 5 → arrays, loop execution
```

These should be merged into a smaller final set:

```text
Arrays
While-loop traversal
Code tracing
```

### Planned Merge Strategy

- normalise topic names
- compare topic embeddings
- merge highly similar candidates
- preserve source chunk IDs
- combine confidence evidence
- avoid losing distinct subtopics

---

## 12. Module 4 — Chapter Identification

### Objective

Identify the most likely broad AQA syllabus chapter for every retained rough topic.

### Approach

1. Store official AQA chapters in the knowledge base.
2. Embed each official chapter description.
3. Build a query from the rough topic.
4. Retrieve top candidate chapters.
5. Apply confidence thresholds.
6. Use an LLM only when results are ambiguous.

---

## 13. Module 5 — Topic and Subtopic Identification

After the chapter is selected:

1. retrieve candidate topics inside the chapter
2. retrieve candidate subtopics
3. rank using embedding similarity
4. use keyword support
5. return top candidates with confidence
6. use LLM fallback only for close results

---

## 14. Module 6 — Official AQA Syllabus Mapping

### Rules

- official syllabus data is the source of truth
- no syllabus label may be generated from memory
- every final mapping must reference a stored official entry
- low-confidence results must be marked uncertain
- ambiguous mappings may use LLM fallback
- no-match topics must remain unmapped rather than forced

### Example Final Mapping

```json
{
  "rough_topic": "array traversal using a while loop",
  "chapter": "Programming",
  "topic": "Programming concepts",
  "subtopic": "Iteration and arrays",
  "confidence": 0.91,
  "official_reference_id": "AQA-8525-3.2.X"
}
```

---

# Agent 2 — Assessment Retrieval

## 15. Agent 2 Objective

Agent 2 receives structured topic mappings from Agent 1 and retrieves relevant AQA past-paper questions and mark-scheme content.

### Input

```json
{
  "subject": "Computer Science",
  "syllabus_topic": "Programming concepts",
  "syllabus_subtopic": "Iteration and arrays"
}
```

### Output

```json
{
  "questions": [
    {
      "question_text": "...",
      "marks": 4,
      "paper": "June 2024 Paper 1B",
      "question_number": "05.2",
      "mark_scheme_excerpt": "...",
      "retrieval_score": 0.89
    }
  ]
}
```

---

## 16. Agent 2 Knowledge Base

The knowledge base will include:

- AQA GCSE Computer Science question papers
- official mark schemes
- specimen papers
- specification documents
- question metadata
- topic tags
- exam session and paper information

### Storage Plan

```text
data/
├── specifications/
├── question_papers/
├── mark_schemes/
└── specimen_material/
```

Processed records will include:

- extracted question text
- question number
- mark value
- paper identifier
- exam year/session
- mark-scheme text
- topic/chapter metadata
- embeddings

---

## 17. Agent 2 Processing Pipeline

```text
Agent 1 topic mapping
        ↓
Build retrieval query
        ↓
Metadata filtering
        ↓
Vector search
        ↓
Keyword matching
        ↓
Re-ranking
        ↓
Question + mark-scheme pairing
        ↓
Assessment pack
```

---

## 18. Quiz Generation Fallback

If no suitable past-paper coverage is found:

```text
No strong retrieved question
        ↓
Generate syllabus-grounded quiz
        ↓
Validate against official syllabus context
```

The quiz-generation LLM should only use retrieved official syllabus content and approved topic mappings.

---

# Technical Stack

## 19. Core Technologies

### Backend

```text
FastAPI
Python
Pydantic
```

### Embeddings

```text
Module 2:
sentence-transformers/all-MiniLM-L6-v2

Mapping and retrieval:
Qwen/Qwen3-Embedding-0.6B
```

### LLM

```text
GPT-OSS-120B through Groq
```

Used only for uncertainty and fallback cases.

### Database / Vector Storage

```text
PostgreSQL
pgvector or Qdrant
```

The final vector-database choice will be confirmed during retrieval testing.

### Document Processing

```text
PyMuPDF for PDF extraction
python-docx for DOCX extraction
```

---

# Proposed Repository Structure

## 20. Project Structure

```text
EDTECH/
├── Agent_1/
│   ├── app/
│   │   ├── schemas/
│   │   │   ├── transcript.py
│   │   │   ├── chunk.py
│   │   │   └── topic.py
│   │   ├── services/
│   │   │   ├── transcript_preprocessor.py
│   │   │   ├── semantic_chunker.py
│   │   │   ├── embedding_service.py
│   │   │   ├── topic_extractor.py
│   │   │   ├── cs_relevance_filter.py
│   │   │   ├── topic_merger.py
│   │   │   └── syllabus_mapper.py
│   │   └── api/
│   ├── scripts/
│   ├── tests/
│   ├── requirements.txt
│   └── README.md
│
├── Agent_2/
│   ├── app/
│   │   ├── schemas/
│   │   ├── services/
│   │   │   ├── document_extractor.py
│   │   │   ├── question_chunker.py
│   │   │   ├── assessment_indexer.py
│   │   │   ├── assessment_retriever.py
│   │   │   ├── reranker.py
│   │   │   └── quiz_generator.py
│   │   └── api/
│   ├── scripts/
│   ├── tests/
│   └── requirements.txt
│
├── shared/
│   ├── database/
│   ├── config/
│   └── schemas/
│
├── .gitignore
├── plan.md
└── README.md
```

---

# Testing Strategy

## 21. Testing Principles

Testing will focus on:

- important architectural decisions
- tool comparison
- failure cases
- realistic transcript noise
- mapping accuracy
- retrieval relevance

The project will avoid testing every trivial function while keeping regression tests for previously discovered bugs.

---

## 22. Current Project Status

### Completed

- [x] Two-agent architecture
- [x] Module 1 deterministic preprocessing
- [x] Timestamp and speaker-label support
- [x] Artefact and filler handling
- [x] Repeated word and sentence handling
- [x] Cross-line duplicate sentence removal
- [x] Module 1 regression tests
- [x] Module 2 semantic chunking
- [x] MiniLM integration
- [x] Adaptive threshold logic
- [x] Strong and soft transition logic
- [x] Maximum chunk-size guardrail
- [x] Selective overlap
- [x] Five-transcript batch testing
- [x] Module 2 regression tests
- [x] Module 1 and Module 2 finalisation

### In Progress / Next

- [ ] Define topic-candidate schema
- [ ] Build rough topic extractor
- [ ] Build Computer Science relevance filter
- [ ] Test multiple topic extraction from one chunk
- [ ] Merge duplicate topics across chunks
- [ ] Prepare official AQA syllabus data
- [ ] Build chapter/topic/subtopic embedding index
- [ ] Implement official syllabus mapping

### Later Work

- [ ] Download and organise AQA past papers
- [ ] Download and organise mark schemes
- [ ] Extract questions and mark-scheme sections
- [ ] Create question/mark-scheme records
- [ ] Generate retrieval embeddings
- [ ] Build Agent 2 hybrid retrieval
- [ ] Implement re-ranking
- [ ] Implement quiz-generation fallback
- [ ] Build FastAPI endpoints
- [ ] Add database persistence
- [ ] Integrate both agents
- [ ] Add end-to-end evaluation
- [ ] Prepare deployment plan

---

# Development Rules

## 23. Rules to Follow

1. Do not use an LLM when deterministic logic or embeddings are sufficient.
2. Do not rewrite transcript meaning during preprocessing.
3. Do not assume one chunk contains one topic.
4. Extract rough topics before syllabus mapping.
5. Filter non-CS topics after extraction, not during cleaning.
6. Preserve source chunk IDs for traceability.
7. Do not invent official AQA labels or references.
8. Return uncertainty when confidence is low.
9. Keep models task-specific.
10. Add regression tests whenever a real failure is discovered.
11. Avoid tuning the pipeline for only one transcript.
12. Keep private lesson transcripts and test data out of the public repository.

---

# Final Planned Pipeline

```text
Lesson transcript
        ↓
Deterministic preprocessing
        ↓
Guarded semantic chunking
        ↓
Rough topic extraction
        ↓
Computer Science relevance filtering
        ↓
Topic merging
        ↓
Official AQA syllabus mapping
        ↓
Past-paper and mark-scheme retrieval
        ↓
Assessment pack
        ↓
Quiz-generation fallback when required
```
