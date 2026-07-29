# EDTech AI — Agent 1

Agent 1 processes lesson transcripts and extracts syllabus-aligned topics.

## Setup

```powershell
cd Agent_1

python -m venv ..\.venv
..\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file inside `Agent_1`:

```env
DATABASE_URL=postgresql+psycopg://postgres:YOUR_PASSWORD@localhost:5432/edtech
GROQ_API_KEY=YOUR_GROQ_API_KEY
GROQ_TECH_CORRECTION_MODEL=openai/gpt-oss-120b
```

## Initialise PostgreSQL

```powershell
python -m scripts.init_technical_corrections_db
```

## Run the Complete Pipeline

```powershell
python -m scripts.run_agent1_pipeline `
  --file "test_data\Transcript 1.docx"
```

Run without Groq:

```powershell
python -m scripts.run_agent1_pipeline `
  --file "test_data\Transcript 1.docx" `
  --no-llm
```

## Pipeline

```text
DOCX transcript
→ preprocessing and cleaning
→ technical terminology normalisation
→ semantic chunking
→ topic extraction and AQA mapping
```

## Outputs

Pipeline outputs are saved in:

```text
test_outputs/pipeline_runs/
```

Each transcript run contains:

```text
01_preprocessing/
├── cleaned_transcript.txt
├── preprocessing_audit.json
└── technical_normalisation_audit.json

02_chunking/
├── chunks.json
└── chunks_readable.txt

03_topic_extraction/
├── topics.json
└── topics_readable.txt

pipeline_manifest.json
```

## Run Technical Normalisation Only

```powershell
python -m scripts.run_selective_technical_normalisation `
  --file "test_data\transcript_1_cleaned.txt"
```

## Review Stored Corrections

List stored corrections:

```powershell
python -m scripts.review_technical_corrections list
```

Approve a correction:

```powershell
python -m scripts.review_technical_corrections approve RECORD_ID
```

Reject a correction:

```powershell
python -m scripts.review_technical_corrections reject RECORD_ID
```

## Tests

```powershell
python -m scripts.test_technical_correction_postgres
python -m scripts.test_selective_technical_normalisation
python -m scripts.test_technical_normalisation_precision
python -m scripts.test_module_2_segment_metadata
python -m scripts.test_module_3_regressions
```

Run Module 3 batch evaluation:

```powershell
python -m scripts.test_module_3_batch `
  --input-dir "test_outputs\module_1_2_batch" `
  --output-dir "test_outputs\module_3_batch_baseline"
```

Run the multi-transcript DOCX integration test:

```powershell
python -m scripts.test_agent1_two_transcripts_docx `
  --files `
  "test_data\Transcript 1.docx" `
  "test_data\Transcript_Raw_3_Algorithms_Programming.docx" `
  --output-root "test_outputs\agent1_two_transcripts_test" `
  --no-llm
```

## What Agent 1 Does

- Extracts text from DOCX transcripts
- Removes timestamps, speaker labels, fillers and transcript noise
- Normalises spoken code such as `array dot length` → `array.length`
- Detects suspicious technical phrases
- Uses PostgreSQL correction memory
- Calls Groq only for uncertain technical corrections
- Creates semantic chunks with continuation metadata
- Extracts lesson topics
- Maps topics to official AQA concepts
- Saves every module output for later use

## Important

Do not commit:

```text
.env
.venv/
__pycache__/
*.pyc
test_outputs/
```

See `plan.md` for the complete architecture.
