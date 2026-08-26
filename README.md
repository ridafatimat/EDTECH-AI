EDTech AI — AI-Assisted Lesson Analysis & Assessment

EDTech AI is an end-to-end assessment pipeline for AQA GCSE Computer Science (8525). It converts lesson transcripts into syllabus-aligned topics and then uses those approved topics to retrieve official-style assessment material — or generate new AQA-aligned questions when coverage is missing.

Tech Stack
Layer	Technology
Frontend	Next.js · React · TypeScript · pnpm
Backend	FastAPI
Orchestration	LangGraph (+ PostgreSQL checkpoints)
Tooling	Model Context Protocol (MCP)
Persistence	PostgreSQL (state + human-review memory)
Retrieval	Qdrant (semantic syllabus & assessment search)
End-to-End Flow
Lesson transcript
Agent 1
Preprocessing & cleaning
Technical terminologynormalisation
Semantic chunking
Topic extraction
AQA syllabus mapping
Human review / HITL
Approved topics
Agent 2
Official question retrieval
Coverage / mark shortfalldetection
AI generation whenrequired
Visual rendering
Question / quiz review
Final question paper +marking scheme PDF
Architecture
edtech_frontend/ — Next.js/ pnpm
api/ — FastAPI
streamlit_integration/ —current runtime bridge
langgraph_orchestration/ —LangGraph workflow +PostgreSQL checkpoints
mcp_server/
Agent_1/ — transcriptprocessing, topic extraction,AQA syllabus mapping
Agent2/ — questionretrieval, quiz generation,visual rendering, final PDF

PostgreSQL

workflow checkpoints
technical-correction memory
HITL decisions
syllabus metadata

Qdrant

syllabus semantic search
assessment-question retrieval
Repository Structure
text
EDTECH-AI/
├── Agent_1/                   # Transcript processing and syllabus mapping
├── Agent2/                    # Retrieval, quiz generation and visual tooling
├── api/                       # FastAPI backend
├── edtech_frontend/           # Next.js / pnpm frontend
├── langgraph_orchestration/   # LangGraph graph and checkpointing
├── mcp_server/                # MCP server, tools, schemas and adapters
├── orchestration/             # Controller, planning, state and guardrails
├── streamlit_integration/     # Current runtime bridge / compatibility layer
├── alembic/                   # Database migrations
├── tests/                     # Integration and orchestration tests
├── requirements_langgraph.txt
├── requirements_mcp.txt
└── README.md
Agent 1 — Transcript to AQA Topics

Agent 1 processes lesson transcripts and determines which AQA GCSE Computer Science concepts were genuinely taught.

Pipeline
Transcript
Preprocessing
Technical normalisation
Semantic chunking
Topic candidate extraction
Evidence-quality evaluation
AQA syllabus mapping
Human review
Approved topics
What Agent 1 Does
Extracts lesson transcript content
Removes timestamps, speaker labels, fillers and transcript noise
Normalises spoken technical / code terminology
Detects suspicious technical phrases
Uses PostgreSQL-backed correction memory
Calls an LLM only when uncertain technical correction requires additional reasoning
Creates semantic chunks with continuation metadata
Extracts Computer Science topic candidates
Evaluates whether the evidence is strong enough to represent a genuinely taught topic
Maps lesson topics to official AQA GCSE Computer Science concepts
Detects CS content that may not map directly to the stored syllabus
Supports human topic correction and editing
Stores reusable HITL decisions
Runtime Code

Core Python services live under:

text
Agent_1/app/

The main Agent 1 notebooks currently live under:

text
Agent_1/Agent1_Streamlit_Frontend/Notebooks/

Important notebooks include:

text
Module1 Preprocessing.ipynb
Module2 Chunking.ipynb
Module3 Topic Mapping.ipynb
AQA Syllabus Storage

The current syllabus runtime uses PostgreSQL + Qdrant rather than the earlier file-based catalogue approach.

Structured syllabus metadata is managed through:

text
Agent_1/app/services/syllabus_store.py

PostgreSQL stores structured concept metadata, while Qdrant provides semantic nearest-neighbour search over syllabus concepts.

Human-in-the-Loop (HITL)

Important decisions are surfaced to the user instead of being silently made by the agents. Examples include:

Approving or correcting topic mappings
Changing a detected topic role
Replacing a topic
Removing a topic
Adding a missing taught topic
Reviewing historical HITL decisions
Approving Agent 2 topics
Reviewing retrieval relevance
Approving, editing, regenerating or rejecting generated questions
Approving or rejecting the final generated quiz

Where applicable, previous human decisions can be stored in PostgreSQL and reused during later runs.

Agent 2 — Assessment Retrieval and Generation

Agent 2 uses the approved Agent 1 topics to build assessment material. The two main assessment paths are:

Approved topics
Official question retrieval
Complete AI quizgeneration
Official Question Retrieval

Retrieval is handled by:

text
Agent2/Notebooks/05_agent1_topics_to_ranked_assessment_retrieval.ipynb

The retrieval flow:

Approved topics
Qdrant question retrieval
Topic / paper filtering
Re-ranking
Retrieval HITL feedback
Approved retrievedquestions

The system preserves associated assessment metadata and marking guidance where available.

Retrieval Shortfall Handling

Official retrieval may not always satisfy the required mark total or topic coverage.

text
Requested:            Retrieved:
5 questions           5 questions
20 marks              15 marks   ← 5-mark shortfall

The pipeline detects the missing coverage and can generate only the required shortfall:

Official retrieved questions
Final hybrid assessment
AI-generated missingcoverage
AI Quiz Generation

Agent 2 supports complete quiz generation as well as generation for retrieval shortfalls.

Important quiz notebooks:

text
Agent2/Notebooks/06_quiz_generation.ipynb
Agent2/Notebooks/06B_quiz_generation.ipynb
Agent2/Notebooks/06C_quiz_generation.ipynb
Quiz Generation Strategies

Plan A — Per-Question Generation

Questions are generated independently.

text
Question 1 → model call
Question 2 → model call
Question 3 → model call
...

Simple to regenerate individual questions, but requires more API calls and repeated prompt tokens.

Plan B — Consolidated Generation

Multiple questions are generated in a single consolidated model request.

Quiz plan
Single consolidatedgeneration call
Questions + markingguidance

Reduces repeated prompt overhead and API-call count.

Plan C — Hybrid Optimisation

Plan C combines both approaches — preserving token/API-call savings while retaining a fallback path when validation fails.

Yes
No
Try consolidated generation
Validate output
Valid?
Accept
Targeted fallback
Supported Quiz Models

Model configuration is stored in:

text
Agent2/Config/quiz_model_config.json

Current providers include Google Gemini, Groq, and OpenAI — with options such as Gemini, GPT-OSS via Groq, and OpenAI GPT models. Only the API key for the provider being used needs to be configured.

Visual Question Handling

Agent 2 supports questions containing or requiring visual material such as:

Logic gate diagrams
Truth tables
Code blocks
Structured tables
Retrieved examination figures
Multi-page question material

Important visual notebooks:

text
Agent2/Notebooks/07_question_visual_cropping_and_multipage_rendering.ipynb
Agent2/Notebooks/08_visual_generation_tool_layer.ipynb

Notebook 08 is exposed through the MCP visual tool layer for generated assets.

Final Assessment Output

The final assessment can contain:

Official retrieved questions
AI-generated AQA-aligned practice questions
Official marking guidance
AI-generated marking guidance
Diagrams and other visual assets
AQA syllabus references
Paper 1 / Paper 2 metadata
Primary / supporting topic roles

The output is combined into a final question paper + marking scheme PDF once the required review state is satisfied.

MCP Layer

The Model Context Protocol layer lives under mcp_server/ and exposes structured tools for:

Agent 1 execution
Agent 1 HITL actions
Agent 2 retrieval
Quiz generation
Visual generation
Final PDF creation

Important directories:

text
mcp_server/adapters/
mcp_server/schemas/
mcp_server/tools/
LangGraph Orchestration

LangGraph coordinates the persisted workflow (langgraph_orchestration/) and is responsible for:

Workflow state
Routing
MCP execution
Human interrupts
Workflow resume behaviour
PostgreSQL checkpoints

The graph is designed so that human-only decisions are not originated automatically by the agent.

Controller Layer

The orchestration controller lives under orchestration/ and manages:

Current workflow state
Assessment intent
Valid next actions
Controller planning
State resolution
Workflow guardrails
Backend API

The current backend entry point is api/main.py. The FastAPI layer connects the Next.js frontend to the persisted LangGraph/MCP workflow, providing endpoints for:

Transcript upload
Run creation and progress
Preprocessing output
Semantic chunking output
Topic mapping
HITL review
Approved topics
Assessment configuration
Retrieval
Quiz generation
Question / quiz review
Generated assessment assets
Final PDF access
Frontend

The user-facing frontend lives under edtech_frontend/ and uses Next.js, React, TypeScript, and pnpm.

It defaults to http://localhost:8000, which can be overridden with:

bash
NEXT_PUBLIC_API_URL=http://localhost:8000
Getting Started
Prerequisites

Before running the full system, install and configure:

Python 3.11+
Node.js
pnpm
PostgreSQL
Qdrant

The project has primarily been developed and tested on Windows / PowerShell.

Clone the Repository
bash
git clone https://github.com/ridafatimat/EDTECH-AI.git
cd EDTECH-AI
Python Environment Setup

Create a virtual environment from the repository root:

powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
Install Python Dependencies
powershell
# Agent 1 runtime dependencies
pip install -r "Agent_1\Agent1_Streamlit_Frontend\frontend\requirements.txt"

# Agent 2 dependencies
pip install -r "Agent2\requirements_agent2.txt"

# MCP dependencies
pip install -r requirements_mcp.txt

# LangGraph dependencies
pip install -r requirements_langgraph.txt

# FastAPI server packages
pip install fastapi uvicorn python-multipart

Dependency note: Agent_1/requirements.txt has not yet been consolidated into the final runtime dependency file. Some Agent 1 dependencies are still stored in the earlier frontend-era requirements file shown above.

Environment Variables

Create a root .env file for local development:

dotenv
DATABASE_URL=postgresql+psycopg://postgres:YOUR_PASSWORD@localhost:5432/edtech

QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=aqa_gcse_computer_science_8525

GROQ_API_KEY=YOUR_GROQ_API_KEY
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
GEMINI_API_KEY=YOUR_GEMINI_API_KEY

Only configure provider keys you intend to use. Optional frontend override in edtech_frontend/.env.local:

dotenv
NEXT_PUBLIC_API_URL=http://localhost:8000

Do not commit real API keys or passwords.

Start Required Services

Before running the application, ensure that:

PostgreSQL is running and the configured EDTech database is available
Qdrant is running and accessible at the configured URL
The required syllabus / question collections have been initialised for the environment
Start the FastAPI Backend

From the repository root:

bash
python -m uvicorn api.main:app --reload --port 8000

The backend will be available at http://localhost:8000.

Start the pnpm Frontend

In a second terminal:

bash
cd edtech_frontend
pnpm install
pnpm dev

The Next.js application is the current user-facing interface.

Runtime Outputs and Diagnostics

Current run artifacts are written under:

text
Agent_1/Agent1_Streamlit_Frontend/runs/

Depending on the workflow stage, a run can contain uploaded transcript data, preprocessing output, semantic chunking output, topic-mapping output, pipeline manifests, HITL state, assessment state, retrieval / generation outputs, diagnostic JSON / CSV information, and final assessment assets.

Generated runtime artifacts should not be committed to source control.

Streamlit Migration Status

The Next.js / pnpm frontend has replaced Streamlit as the primary user-facing UI. However, not every Streamlit-named directory can be deleted yet — the backend still references runtime paths and bridge code under:

text
Agent_1/Agent1_Streamlit_Frontend/
streamlit_integration/

These directories currently contain a mixture of legacy Streamlit UI code, runtime bridge logic, notebook locations, and run / diagnostic path compatibility.

Intended cleanup path:

Separate runtime and diagnostic logic from UI-specific Streamlit code
Move required runtime components into neutral backend / orchestration locations
Update imports and paths
Verify the project from a fresh clone
Then remove the remaining obsolete Streamlit UI components

These directories should not be removed blindly while they remain part of the current execution path.

Tests

Run the main test suite with:

bash
python -m pytest tests

Additional Agent 1 diagnostic and migration utilities are available under Agent_1/scripts/.

Git Branches

main — the cleaner, execution-focused version of the project, including Agent 1 execution logic, Agent 2 retrieval and generation, the FastAPI backend, the pnpm / Next.js frontend, the MCP server and adapters, LangGraph orchestration, and required runtime bridge components.

backup/pre-cleanup-migration — preserves the pre-cleanup development state and historical implementation files. Intended as a recovery / reference branch; it should not be merged directly into main.

Do Not Commit

Do not commit local secrets, environments, caches or generated runtime outputs:

gitignore
.env
.env.*
.venv/
node_modules/
.next/
__pycache__/
*.pyc
.pytest_cache/
runs/
OUTPUT/
test_outputs/
Project Goal

EDTech AI aims to provide a traceable lesson-to-assessment workflow for AQA GCSE Computer Science where:

Topics are grounded in what was actually taught
Syllabus mappings can be reviewed and corrected
Human decisions remain explicit at important gates
Official assessment material is retrieved when suitable
AI generation fills genuine assessment gaps
Token / API usage can be optimised without removing validation safeguards
Visual questions and marking schemes remain aligned
The final assessment remains auditable through persisted run outputs and diagnostics
