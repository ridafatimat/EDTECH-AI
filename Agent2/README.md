# Agent 2 Knowledge Base — Starter

1. Copy `01_aqa_source_discovery_and_registration.ipynb` into `Agent_2/notebooks/`.
2. Copy `.env.example` to `Agent_2/.env` and add the PostgreSQL password.
3. Activate the project virtual environment.
4. Install dependencies:
   `pip install -r requirements_agent2.txt`
5. Install Chromium:
   `playwright install chromium`
6. Run the notebook from top to bottom.

The notebook creates/uses:
- `assessment_documents`
- `assessment_ingestion_runs`
- `cache/assessment_pdfs`
- `OUTPUT/` manifests
