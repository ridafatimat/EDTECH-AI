EDTech AI — Agent 1

Agent 1 processes lesson transcripts and extracts syllabus-aligned topics.

Setup

cd Agent_1

python -m venv ..\.venv
..\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

Create Agent_1/.env:

DATABASE_URL=postgresql+psycopg://postgres:YOUR_PASSWORD@localhost:5432/edtech
GROQ_API_KEY=YOUR_GROQ_API_KEY
GROQ_TECH_CORRECTION_MODEL=openai/gpt-oss-120b

Initialise PostgreSQL

python -m scripts.init_technical_corrections_db

Run the complete pipeline

python -m scripts.run_agent1_pipeline `
  --file "test_data\Transcript 1.docx"

Run without Groq:

python -m scripts.run_agent1_pipeline `
  --file "test_data\Transcript 1.docx" `
  --no-llm

Pipeline

DOCX transcript
→ preprocessing and cleaning
→ technical terminology normalisation
→ semantic chunking
→ topic extraction and AQA mapping

Outputs

Outputs are saved in:

test_outputs/pipeline_runs/

Each transcript run contains:

01_preprocessing/cleaned_transcript.txt
02_chunking/chunks.json
03_topic_extraction/topics.json
pipeline_manifest.json

Run technical normalisation only

python -m scripts.run_selective_technical_normalisation `
  --file "test_data\transcript_1_cleaned.txt"

Review stored corrections

python -m scripts.review_technical_corrections list
python -m scripts.review_technical_corrections approve RECORD_ID
python -m scripts.review_technical_corrections reject RECORD_ID

Tests

python -m scripts.test_technical_correction_postgres
python -m scripts.test_selective_technical_normalisation
python -m scripts.test_technical_normalisation_precision
python -m scripts.test_module_2_segment_metadata
python -m scripts.test_module_3_regressions

Run Module 3 batch evaluation:

python -m scripts.test_module_3_batch `
  --input-dir "test_outputs\module_1_2_batch" `
  --output-dir "test_outputs\module_3_batch_baseline"

What Agent 1 does

Extracts text from DOCX transcripts

Removes timestamps, speaker labels, fillers and transcript noise

Normalises spoken code such as array dot length → array.length

Uses PostgreSQL correction memory

Calls Groq only for uncertain technical phrases

Creates semantic chunks with continuation metadata

Extracts lesson topics and maps them to AQA concepts

Saves every module output for later use

Important

Do not commit:

.env
.venv/
__pycache__/
*.pyc
test_outputs/

See plan.md for the complete architecture.
