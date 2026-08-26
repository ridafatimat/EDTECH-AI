EDTech AI — AI-Assisted Lesson Analysis & Assessment

EDTech AI is an end-to-end assessment pipeline for AQA GCSE Computer Science (8525). It converts lesson transcripts into syllabus-aligned topics and then uses those approved topics to retrieve official-style assessment material or generate new AQA-aligned questions when coverage is missing.

The current application uses a Next.js / pnpm frontend, a FastAPI backend, LangGraph for workflow orchestration, Model Context Protocol (MCP) for tool execution, PostgreSQL for persistent state and human-review memory, and Qdrant for semantic syllabus and assessment retrieval.

End-to-End Flow

Lesson transcript
        ↓
Agent 1
        ↓
Preprocessing and cleaning
        ↓
Technical terminology normalisation
        ↓
Semantic chunking
        ↓
Topic extraction
        ↓
AQA syllabus mapping
        ↓
Human review / HITL
        ↓
Approved topics
        ↓
Agent 2
        ↓
Official question retrieval
        ↓
Coverage / mark shortfall detection
        ↓
AI generation when required
        ↓
Visual rendering
        ↓
Question / quiz review
        ↓
Final question paper + marking scheme PDF

Architecture

edtech_frontend/
       │
       │  Next.js / pnpm
       ▼
api/
       │
       │  FastAPI
       ▼
streamlit_integration/
       │
       │  Current runtime bridge
       ▼
langgraph_orchestration/
       │
       │  LangGraph workflow + PostgreSQL checkpoints
       ▼
mcp_server/
       │
       ├──────────────► Agent_1/
       │                 Transcript processing
       │                 Topic extraction
       │                 AQA syllabus mapping
       │
       └──────────────► Agent2/
                         Question retrieval
                         Quiz generation
                         Visual rendering
                         Final PDF generation

PostgreSQL
    ├── workflow checkpoints
    ├── technical-correction memory
    ├── HITL decisions
    └── syllabus metadata

Qdrant
    ├── syllabus semantic search
    └── assessment-question retrieval

Repository Structure

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

Agent 1 Pipeline

Transcript
    ↓
Preprocessing
    ↓
Technical normalisation
    ↓
Semantic chunking
    ↓
Topic candidate extraction
    ↓
Evidence-quality evaluation
    ↓
AQA syllabus mapping
    ↓
Human review
    ↓
Approved topics

Agent 1 currently supports:

transcript text extraction

removal of timestamps, speaker labels, fillers and transcript noise

spoken technical/code normalisation

detection of suspicious technical phrases

PostgreSQL-backed correction memory

LLM fallback for uncertain technical corrections

semantic chunking with continuation metadata

topic candidate extraction

evidence-quality filtering

AQA syllabus mapping

detection of relevant CS content that does not map cleanly

human topic correction and editing

reusable HITL memory

Agent 1 Runtime Code

Core Python services live under:

Agent_1/app/

The main Agent 1 notebooks currently live under:

Agent_1/Agent1_Streamlit_Frontend/Notebooks/

Important notebooks include:

Module1 Preprocessing.ipynb
Module2 Chunking.ipynb
Module3 Topic Mapping.ipynb

AQA Syllabus Storage

The current syllabus runtime uses PostgreSQL + Qdrant rather than the earlier file-based catalogue approach.

Structured syllabus metadata is managed through:

Agent_1/app/services/syllabus_store.py

PostgreSQL stores the concept metadata and Qdrant provides semantic nearest-neighbour search over syllabus concepts.

Human-in-the-Loop (HITL)

Important decisions are explicitly surfaced to the user instead of being silently made by the agents.

Examples include:

approving or correcting topic mappings

changing a detected topic role

replacing a topic

removing a topic

adding a missing taught topic

reviewing historical HITL decisions

approving Agent 2 topics

reviewing retrieval relevance

approving, editing, regenerating or rejecting generated questions

approving or rejecting the final generated quiz

Where applicable, previous human decisions can be stored in PostgreSQL and reused during later runs.

Agent 2 — Assessment Retrieval and Generation

Agent 2 uses the approved Agent 1 topics to build assessment material.

The two main assessment paths are:

Approved topics
      │
      ├──────────────► Official question retrieval
      │
      └──────────────► Complete AI quiz generation

Official Question Retrieval

Retrieval is handled by:

Agent2/Notebooks/05_agent1_topics_to_ranked_assessment_retrieval.ipynb

The retrieval flow includes:

Approved topics
      ↓
Qdrant question retrieval
      ↓
Topic / paper filtering
      ↓
Re-ranking
      ↓
Retrieval HITL feedback
      ↓
Approved retrieved questions

The system preserves associated assessment metadata and marking guidance where available.

Retrieval Shortfall Handling

Official retrieval may not always satisfy the required mark total or topic coverage.

Example:

Requested:
5 questions
20 marks

Retrieved:
5 questions
15 marks

The pipeline detects the missing coverage and can generate only the required shortfall:

Official retrieved questions
            +
AI-generated missing coverage
            ↓
Final hybrid assessment

AI Quiz Generation

Agent 2 supports complete quiz generation as well as generation for retrieval shortfalls.

Important quiz notebooks are:

Agent2/Notebooks/06_quiz_generation.ipynb
Agent2/Notebooks/06B_quiz_generation.ipynb
Agent2/Notebooks/06C_quiz_generation.ipynb

Quiz Generation Strategies

Plan A — Per-Question Generation

Questions are generated independently.

Question 1 → model call
Question 2 → model call
Question 3 → model call
...

This makes individual question regeneration simple, but requires more API calls and repeated prompt tokens.

Plan B — Consolidated Generation

Multiple questions are generated in a consolidated model request.

Quiz plan
    ↓
Single consolidated generation call
    ↓
Questions + marking guidance

This reduces repeated prompt overhead and API-call count.

Plan C — Hybrid Optimisation

Plan C combines the two approaches.

Try consolidated generation
            ↓
Validate output
            ↓
        Valid?
        /    \
      Yes     No
       ↓       ↓
    Accept   Targeted fallback

This is designed to preserve the token/API-call savings of consolidated generation while retaining a fallback path when validation fails.

Supported Quiz Models

Model configuration is stored in:

Agent2/Config/quiz_model_config.json

The current configuration includes providers such as:

Google Gemini

Groq

OpenAI

Configured model options currently include Gemini, GPT-OSS via Groq, and OpenAI GPT models.

Only the API key for the provider being used needs to be configured.

Visual Question Handling

Agent 2 supports questions containing or requiring visual material such as:

logic gate diagrams

truth tables

code blocks

structured tables

retrieved examination figures

multi-page question material

Important visual notebooks include:

Agent2/Notebooks/07_question_visual_cropping_and_multipage_rendering.ipynb
Agent2/Notebooks/08_visual_generation_tool_layer.ipynb

Notebook 08 is exposed through the MCP visual tool layer for generated assets.

Final Assessment Output

The final assessment can contain:

official retrieved questions

AI-generated AQA-aligned practice questions

official marking guidance

AI-generated marking guidance

diagrams and other visual assets

AQA syllabus references

Paper 1 / Paper 2 metadata

primary / supporting topic roles

The output is combined into a final question paper + marking scheme PDF after the required review state is satisfied.

MCP Layer

The Model Context Protocol layer lives under:

mcp_server/

It exposes structured tools for:

Agent 1 execution

Agent 1 HITL actions

Agent 2 retrieval

quiz generation

visual generation

final PDF creation

Important directories include:

mcp_server/adapters/
mcp_server/schemas/
mcp_server/tools/

LangGraph Orchestration

LangGraph coordinates the persisted workflow.

The implementation lives under:

langgraph_orchestration/

It is responsible for:

workflow state

routing

MCP execution

human interrupts

workflow resume behaviour

PostgreSQL checkpoints

The graph is designed so that human-only decisions are not originated automatically by the agent.

Controller Layer

The orchestration controller lives under:

orchestration/

It manages:

current workflow state

assessment intent

valid next actions

controller planning

state resolution

workflow guardrails

Backend API

The current backend entry point is:

api/main.py

The FastAPI layer connects the Next.js frontend to the persisted LangGraph/MCP workflow.

It provides endpoints for operations including:

transcript upload

run creation and progress

preprocessing output

semantic chunking output

topic mapping

HITL review

approved topics

assessment configuration

retrieval

quiz generation

question / quiz review

generated assessment assets

final PDF access

Frontend

The current user-facing frontend lives under:

edtech_frontend/

It uses:

Next.js

React

TypeScript

pnpm

The frontend defaults to the backend URL:

http://localhost:8000

This can be overridden using:

NEXT_PUBLIC_API_URL=http://localhost:8000

Prerequisites

Before running the full system, install and configure:

Python 3.11+

Node.js

pnpm

PostgreSQL

Qdrant

The project has primarily been developed and tested on Windows / PowerShell.

Clone the Repository

git clone https://github.com/ridafatimat/EDTECH-AI.git
cd EDTECH-AI

Python Environment Setup

Create a virtual environment from the repository root:

python -m venv .venv

Activate it:

.\.venv\Scripts\Activate.ps1

Install Python Dependencies

The current Python dependency files are separated by subsystem.

Install the Agent 1 runtime dependencies:

pip install -r "Agent_1\Agent1_Streamlit_Frontend\frontend\requirements.txt"

Install Agent 2 dependencies:

pip install -r "Agent2\requirements_agent2.txt"

Install MCP dependencies:

pip install -r requirements_mcp.txt

Install LangGraph dependencies:

pip install -r requirements_langgraph.txt

Install the FastAPI server packages:

pip install fastapi uvicorn python-multipart

Dependency note: Agent_1/requirements.txt has not yet been consolidated into the final runtime dependency file. Some Agent 1 dependencies are still stored in the earlier frontend-era requirements file shown above. This should be consolidated during the later Streamlit/runtime cleanup.

Environment Variables

Create a root .env file for local development.

Example:

DATABASE_URL=postgresql+psycopg://postgres:YOUR_PASSWORD@localhost:5432/edtech

QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=aqa_gcse_computer_science_8525

GROQ_API_KEY=YOUR_GROQ_API_KEY
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
GEMINI_API_KEY=YOUR_GEMINI_API_KEY

Only configure provider keys you intend to use.

Optional frontend override:

edtech_frontend/.env.local

NEXT_PUBLIC_API_URL=http://localhost:8000

Do not commit real API keys or passwords.

Start Required Services

Before running the application, ensure that:

PostgreSQL is running and the configured EDTech database is available

Qdrant is running and accessible at the configured URL

the required syllabus/question collections have been initialised for the environment being used

Start the FastAPI Backend

From the repository root:

python -m uvicorn api.main:app --reload --port 8000

The backend will be available at:

http://localhost:8000

Start the pnpm Frontend

Open a second terminal:

cd edtech_frontend
pnpm install
pnpm dev

The Next.js application is the current user-facing interface.

Runtime Outputs and Diagnostics

Current run artifacts are written under:

Agent_1/Agent1_Streamlit_Frontend/runs/

Depending on the workflow stage, a run can contain:

uploaded transcript data

preprocessing output

semantic chunking output

topic-mapping output

pipeline manifests

HITL state

assessment state

retrieval / generation outputs

diagnostic JSON / CSV information

final assessment assets

Generated runtime artifacts should not be committed to source control.

Current Streamlit Migration Status

The Next.js / pnpm frontend has replaced Streamlit as the primary user-facing UI.

However, not every Streamlit-named directory can be deleted yet.

The current backend still references runtime paths and bridge code under:

Agent_1/Agent1_Streamlit_Frontend/
streamlit_integration/

These directories currently contain a mixture of:

legacy Streamlit UI code

runtime bridge logic

notebook locations

run / diagnostic path compatibility

The intended cleanup path is:

separate runtime and diagnostic logic from UI-specific Streamlit code,

move required runtime components into neutral backend/orchestration locations,

update imports and paths,

verify the project from a fresh clone,

then remove the remaining obsolete Streamlit UI components.

The directories should therefore not be removed blindly while they remain part of the current execution path.

Tests

Run the main test suite with:

python -m pytest tests

Additional Agent 1 diagnostic and migration utilities are available under:

Agent_1/scripts/

Git Branches

main

The main branch contains the cleaner execution-focused version of the project, including:

Agent 1 execution logic

Agent 2 retrieval and generation

FastAPI backend

pnpm / Next.js frontend

MCP server and adapters

LangGraph orchestration

required runtime bridge components

backup/pre-cleanup-migration

This branch preserves the pre-cleanup development state and historical implementation files.

It is intended as a recovery/reference branch and should not be merged directly into main.

Do Not Commit

Do not commit local secrets, environments, caches or generated runtime outputs.

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

topics are grounded in what was actually taught,

syllabus mappings can be reviewed and corrected,

human decisions remain explicit at important gates,

official assessment material is retrieved when suitable,

AI generation fills genuine assessment gaps,

token/API usage can be optimised without removing validation safeguards,

visual questions and marking schemes remain aligned, and

the final assessment remains auditable through persisted run outputs and diagnostics.
