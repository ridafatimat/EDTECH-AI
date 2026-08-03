# Agent 1 Streamlit Frontend

This testing interface runs the three existing notebooks sequentially for one uploaded transcript:

1. Module 1 — preprocessing
2. Module 2 — semantic chunking
3. Module 3 — AQA topic mapping and selective Groq fallback

The notebooks keep their original batch behaviour. The frontend enables a separate environment-driven single-file mode only while a web request is running.

## Put the files in your project

Copy these folders into your `Agent_1` directory:

```text
Agent_1/
├── Notebooks/
│   ├── Module1 Preprocessing.ipynb
│   ├── Module2 Chunking.ipynb
│   └── Module3 Topic Mapping.ipynb
├── frontend/
│   ├── app.py
│   ├── pipeline_runner.py
│   └── requirements.txt
├── Test Data/
├── OUTPUT/
├── runs/
└── .env
```

Back up your current notebooks before replacing them.

## Install

Open the VS Code terminal from `Agent_1` and activate the same virtual environment used by the notebooks.

```powershell
.\.venv\Scripts\activate
python -m pip install -r frontend\requirements.txt
python -m ipykernel install --user --name agent1-venv --display-name "Agent1 venv"
```

## Qdrant

Module 3 requires Qdrant. For local Docker:

```powershell
docker start qdrant
```

When the container does not exist yet:

```powershell
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

The local dashboard should open at `http://localhost:6333/dashboard`.

Example `.env`:

```env
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
GROQ_API_KEY=your_key_here
```

Groq is used only when selective fallback is required.

## Run the page

From the `Agent_1` directory:

```powershell
python -m streamlit run frontend\app.py
```

Upload a PDF, DOCX, or TXT transcript and select **Run Agent 1 Pipeline**.

## Generated run structure

Each upload is isolated, so previous results are never overwritten:

```text
runs/job_<timestamp>_<id>/
├── input/
├── output/<transcript>/
│   ├── 01_cleaned_transcript.txt
│   ├── 01_preprocessing.json
│   ├── 01_preprocessing.pdf
│   ├── 02_chunking.json
│   ├── 02_chunking.pdf
│   ├── 03_topic_mapping.json
│   ├── 03_topics_readable.pdf
│   ├── 04_llm_mapping.pdf
│   └── 05_final_topic_summary.pdf
├── executed_notebooks/
├── logs/
└── pipeline_manifest.json
```

`03_topic_mapping.json` is an additive machine-readable output used by the frontend. It does not replace or change the three Module 3 PDFs.

## Failure behaviour

If a module fails:

- the pipeline stops before the next module;
- completed upstream outputs remain saved;
- the full notebook error is saved inside the run's `logs` folder;
- a new upload gets a different run folder.
