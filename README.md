# Employee Handbook Assistant

A **retrieval-augmented generation (RAG)** application that answers employees' questions about their organisation using only the official employee handbook as its source of truth.

The application retrieves the most relevant handbook sections, generates accurate, citation-backed answers, and declines to answer questions that aren't covered by the handbook instead of hallucinating information.
## Table of contents

- [Problem](#problem)
- [Demo](#demo)
- [Screenshots](#screenshots)
- [Features](#features)
- [How it works](#how-it-works)
- [Evaluation](#evaluation)
- [Project Dependencies](#project-dependencies)
- [Getting Started](#getting-started)
- [Project structure](#project-structure)
- [Configuration](#configuration)
- [Known limitations](#known-limitations)

---

## Problem

Handbooks are long, duplicated across docs, and rarely indexed for natural-language questions. Employees constantly need to look up policies, holiday entitlement, expenses, parental leave, security rules, and so on. Searching them is slow, and answers drift from what the handbook actually says.

This project solves that with a **grounded Q&A assistant**:

- **No hallucination by design.** The generation prompt instructs the model to answer *only* from retrieved context and to say it doesn't know when the handbook is silent. See `INSTRUCTIONS` in [`src/rag/answer_chain.py`](src/rag/answer_chain.py).
- **Measurable quality.** Every retrieval strategy and every generated answer is scored automatically (see [Evaluation](#evaluation)), so you can compare options rather than guessing.
- **Observable in production.** Latency, token usage, cost, errors, and evaluation scores are written to PostgreSQL and visualised in Grafana.

---

## Demo

![Assistant Demo](https://github.com/user-attachments/assets/febbdf5c-11ca-45a9-b138-c0e93c12f2f3)

---


## Screenshots

**Chat UI** — ask a question, get a grounded answer, and rate it (👍 / 👎):

<p align="center">
  <img src="https://github.com/user-attachments/assets/67cd449a-73df-4b38-a391-4b038ed32fbf" alt="Chat UI">
</p>

**Monitoring dashboard** — Grafana panels tracking message volume, latency, token usage, cost, error rates, and evaluation scores:

<p align="center">
  <img src="https://github.com/user-attachments/assets/8fc58953-b6b6-4d45-aa55-e185119735f7" alt="Chat UI">
</p>

---

## Features

### Retrieval & generation

- **Grounded answers** — generation is restricted to retrieved handbook context with an explicit "I don't know based on the handbook" fallback.
- **Hybrid retrieval + optional reranking** — configurable `α` and reranker toggle via environment variables.
- **Query rewriting** — optional LLM step that normalises informal questions before retrieval.
- **Prompt injection resistance** — user text and retrieved context are treated as data, not instructions, and are sanitised before prompting.

### Evaluation & feedback

- **Evaluation harness** — automatic retrieval scoring (Hit Rate, MRR) and LLM-as-judge scoring of generated answers (Faithfulness, Context Relevance, Completeness).
- **Answer rating** — thumbs up/down feedback stored with each message and correlated with automatic judge scores.

### Operations & observability

- **Chat persistence** — conversations, messages, ratings, and per-answer metrics are stored in PostgreSQL. (The current UI opens a new conversation per browser session; the persisted data is the foundation for a future conversation-history picker.)
- **Observability** — latency, token usage, cost, and errors recorded for every message and surfaced in Grafana.
- **Resilience** — retries with exponential backoff for DB connections and LLM calls, plus a per-minute rate limiter for eval workloads.

---

## How it works

```
GitHub handbook (.md files)
        │
        ▼
Ingestion pipeline ──► all-MiniLM-L6-v2 (ONNX) ──► pgvector (HNSW) + tsvector
(src/ingestion/pipeline.py)                              │
                                                         ▼
Employee question (Streamlit chat UI)                    │
        │                                                │
        ▼                                                │
Query rewriting (optional, LLM)                          │
        │                                                │
        ▼                                                │
Hybrid search:  α × vector score + (1 − α) × keyword score
        │
        ▼
Reranking (optional, cross-encoder)                      │
        │                                                │
        ▼                                                │
Prompt built from retrieved context ─────────────────────┘
        │
        ▼
LLM answer (OpenAI-compatible endpoint)
        │
        ├─► conversations / messages / message_metrics (PostgreSQL)
        ├─► sampled LLM-as-judge evaluation (evaluation_runs/results)
        └─► Grafana dashboard (queries the same PostgreSQL tables)
```

### Data

To follow the project, it helps to know what data it touches. The system uses **five kinds of data**:

1. **The handbook itself** — plain Markdown files (the raw text of every policy, benefit, and guide). The app pulls these down from a public GitHub repository, and they are the *only* source the assistant is allowed to answer from.
2. **Embeddings** — a searchable, numeric copy of the handbook. Before any question can be answered, each document is converted into a *vector* (a long list of numbers that captures its meaning), and these vectors are stored in a special database table so the app can quickly find the documents most similar to a question.
3. **Ground truth** — a test set of real questions an employee might ask, each paired with the handbook document that answers it. This is the "answer key" used to measure how good retrieval is.
4. **Evaluation outputs** — the scores produced when the project tests itself (retrieval Hit Rate / MRR and LLM-as-judge quality scores).
5. **Usage metrics** — the runtime record of every chat message: what was asked, what was answered, how long it took, token/cost usage, and whether the user rated it thumbs up or down.

The table below shows each piece and where it lives:

| Data | Description | Location |
| --- | --- | --- |
| Handbook documents | Markdown files fetched from GitHub, embedded, stored in PostgreSQL | DB table `handbook_documents` (cached copy in `data/employee_handbook_documents.json`) |
| Ground truth | Q&A pairs of the form `(question, document path)` used to score retrieval | `data/ground_truth.csv` |
| Evaluation outputs | Hit Rate / MRR summaries and LLM-judge scores | `data/evaluation/` |
| ONNX models | Embedder + reranker, downloaded to `models/` | `models/Xenova/` |
| Metrics | Conversations, messages, ratings, latency, tokens, cost, errors | DB tables `conversations`, `messages`, `message_metrics`, `error_log` |

---


## Evaluation

The project measures two things: how well retrieval finds the right document, and how good the generated answers are. On the retrieval side, the following search strategies are implemented and compared (see [`src/rag/answer_chain.py`](src/rag/answer_chain.py) and [`src/retrieval/`](src/retrieval/)):

1. **Vector search** — cosine similarity over the `embedding <=>` HNSW index.
2. **Keyword search** — PostgreSQL `ts_rank_cd` over the generated `content_tsv` column.
3. **Hybrid search** — reciprocal-rank fusion, weighted by `α` (vector) vs `1 − α` (keyword). Optionally followed by cross-encoder **reranking**.


### Retrieval evaluation (`evaluate-search`)

Runs every ground-truth question against each retrieval method and reports, at top-k (`NUM_RESULTS`, default 5):

- **Hit Rate @ k** — the fraction of questions for which the expected document appears in the top-k results.
- **Mean Reciprocal Rank (MRR)** — the average of `1 / rank` of the first relevant result. Rewards placing the right document *higher*.

Methods compared: `vector`, `keyword`, `hybrid_0.2`, `hybrid_0.5`, `hybrid_0.8`, and `rerank_hybrid_*` variants when the reranker is available. Results are written to `data/evaluation/evaluation_summary.csv`, `evaluation_debug.csv`, and `evaluation_summary.json`, then printed to the console.

**Example output** (from the checked-in `data/evaluation/evaluation_summary.csv`, 925 questions):

```
method              hit_rate   mrr
rerank_hybrid_0.5   0.521      0.411
rerank_hybrid_0.8   0.521      0.411
rerank_hybrid_0.2   0.521      0.411
hybrid_0.5          0.483      0.357
hybrid_0.8          0.484      0.354
hybrid_0.2          0.482      0.356
vector              0.477      0.352
keyword             0.030      0.028
```

Key takeaways from this run: reranking consistently beats plain hybrid, and pure keyword search performs poorly on natural-language questions.

### Ground-truth generation (`generate-ground-truth`)

A configurable LLM (`LLM_MODEL`) reads each handbook document and writes `NUM_QUESTIONS_PER_DOC` (default 5) natural questions an employee might ask, along with the document path that answers each question. Output goes to `data/ground_truth.csv`.

### Generation evaluation (`evaluate-llm`)

Uses an **LLM-as-judge** (separate `JUDGE_MODEL`) to score every generated answer on three dimensions, each **1–5**:

1. **Faithfulness** — is every claim in the answer supported by the retrieved context?
2. **Context relevance** — is the retrieved context relevant to the question?
3. **Completeness** — does the answer fully address the question?

Judging criteria are defined in `JUDGE_INSTRUCTIONS` in [`src/evaluation/llm_eval.py`](src/evaluation/llm_eval.py). Results are stored in `evaluation_runs` / `evaluation_results` tables and written to `data/evaluation/llm_evaluation_detail.csv` and `llm_evaluation_summary.json`.

### Continuous / online evaluation

The chat app **samples** a fraction of live answers (`EVAL_SAMPLE_RATE`, default `0.1`) and asynchronously scores them with the judge model (`src/evaluation/judge.py`). Combined with the user 👍/👎 ratings stored on each message, this lets you correlate human ratings with automatic scores (`get_correlation()`).

---

## Project Dependencies

### Python runtime (`pyproject.toml`)

| Dependency | Version | Purpose |
| --- | --- | --- |
| Python | >= 3.13 | Runtime, managed with [uv](https://docs.astral.sh/uv/) |
| `streamlit` | 1.59.2 | Chat UI |
| `onnxruntime` | 1.27.0 | CPU inference for the embedder and reranker (no GPU required) |
| `numpy` | 2.4.6 | Vector math for embedding/reranking |
| `tokenizers` | 0.23.1 | Tokenization for the ONNX models |
| `openai` | 2.46.0 | OpenAI-compatible client (generation, rewriting, judging) |
| `psycopg[binary]` | 3.3.4 | PostgreSQL driver |
| `pgvector` | 0.5.0 | pgvector bindings for psycopg |
| `pandas` | 3.0.3 | Ground-truth and evaluation data handling |
| `pydantic` | 2.13.4 | Structured LLM-output schemas (`EvaluationScores`, `Questions`) |
| `python-dotenv` | 1.2.2 | `.env` loading |
| `requests` | 2.34.2 | GitHub API / file fetching during ingestion |
| `huggingface-hub` | 1.24.0 | Model downloads |
| `tqdm` | 4.69.0 | Progress bars in eval/generation scripts |

Dev group: `jupyter` 1.1.1, `pytest` 9.1.1, `pytest-mock` 3.15.1, `pytest-dotenv` 0.5.2, `pytest-cov` 7.1.0.

### Infrastructure services (`docker-compose.yml`)

| Service | Image | Purpose |
| --- | --- | --- |
| App | built from `Dockerfile` (Python 3.13) | Streamlit UI + RAG pipeline |
| `app_postgres` | `pgvector/pgvector:pg18` | Vector store with HNSW indexing and full-text search |
| `kestra` | `kestra/kestra:v1.3.28` | Scheduled/triggered ingestion workflow |
| `kestra_postgres` | `postgres:18` | Kestra's metadata store |
| `grafana` | `grafana/grafana:11.2.2` | Observability dashboard over the metrics tables |

The app connects to any **OpenAI-compatible LLM API** (Gemini via its OpenAI-compatible endpoint, OpenAI, Ollama, vLLM, etc.) for generation, query rewriting, ground-truth generation, and LLM-as-judge evaluation.

---

## Getting Started

To get the project up and running, follow the [**Setup guide**](docs/setup.md). It covers everything you need beforehand — Python 3.13+, PostgreSQL with pgvector, and an OpenAI-compatible LLM endpoint — then walks you through configuring the environment, downloading the models, initialising the database, and starting the stack either locally or with Docker (app, pgvector, Kestra, and Grafana).

Once it's running, the [**Usage guide**](docs/usage.md) explains the workflow: ingesting the handbook, generating ground truth, running evaluations, and chatting with the assistant in the UI — along with example inputs and outputs.

---

## Project structure

```
├── src/
│   ├── main.py                 # CLI entry point (ui | ingest | evaluate-search | evaluate-llm | generate-ground-truth)
│   ├── ingestion/pipeline.py   # Fetch .md from GitHub → embed → store in pgvector
│   ├── retrieval/
│   │   ├── embedder.py         # ONNX sentence embeddings (all-MiniLM-L6-v2)
│   │   ├── vectorstore.py      # pgvector connection, HNSW/ivfflat index, tsvector, init_db
│   │   ├── reranker.py         # ONNX cross-encoder reranker (ms-marco-MiniLM-L-6-v2)
│   │   └── filters.py          # Sanitisation before prompting
│   ├── rag/
│   │   ├── answer_chain.py     # RAG class: hybrid search, prompt building, LLM call, metrics
│   │   └── query_rewriter.py   # Optional LLM query rewriting for retrieval
│   ├── services/
│   │   ├── rag_service.py      # Wires up embedder/LLM/reranker/rewriter/evaluator
│   │   └── chat_store.py       # Conversations, messages, ratings, metrics (async writes)
│   ├── evaluation/
│   │   ├── search_eval.py      # Retrieval eval: Hit Rate, MRR
│   │   ├── llm_eval.py         # Offline LLM-as-judge eval (batch)
│   │   ├── judge.py            # Online sampled LLM-as-judge eval (background worker)
│   │   ├── datasets.py         # Ground-truth question generation
│   │   └── metrics.py          # Error logging / timers
│   ├── domain/                 # Pydantic models (EvaluationScores, JudgeResult, Questions)
│   └── ui/streamlit_app.py     # Chat interface
├── scripts/
│   ├── download_models.py      # Fetch ONNX models from Hugging Face
│   └── init_database.py        # Create DB + run database/init.sql
├── database/init.sql           # Full schema (documents, chats, metrics, eval, errors)
├── flows/ingest-handbook.yaml  # Kestra workflow for ingestion
├── grafana/                    # Provisioned dashboard + datasource
├── docs/                       # Setup and usage guides (+ screenshots)
├── data/                       # Ground truth + evaluation outputs (gitignored)
├── models/                     # Downloaded ONNX models (gitignored)
└── tests/                      # pytest suite
```

---

## Configuration

All configuration is via environment variables (see [`.env.example`](.env.example) and [`docs/setup.md`](docs/setup.md) for the full reference). The essentials:

| Variable | Purpose | Default |
| --- | --- | --- |
| `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` | OpenAI-compatible endpoint for generation | — |
| `JUDGE_BASE_URL` / `JUDGE_API_KEY` / `JUDGE_MODEL` | Judge LLM for answer evaluation (disabled if unset) | — |
| `PGHOST` / `PGPORT` / `PGDATABASE` / `PGUSER` / `PGPASSWORD` | PostgreSQL connection | `localhost:5432` |
| `GITHUB_OWNER` / `GITHUB_REPO` / `GITHUB_BRANCH` | Source handbook to ingest | `madetech/handbook@main` |
| `MODEL_PATH` / `VECTOR_DIM` / `TABLE_NAME` | Embedding model, vector size, storage table | `models/Xenova/all-MiniLM-L6-v2` / `384` / `handbook_documents` |
| `RERANKER_ENABLED` / `RERANKER_MODEL_PATH` | Toggle reranking | `false` |
| `QUERY_REWRITER_ENABLED` / `QUERY_REWRITER_MODEL` | Toggle query rewriting | `false` |
| `HYBRID_ALPHAS` | Alpha values evaluated | `0.2,0.5,0.8` |
| `EVAL_SAMPLE_RATE` | Fraction of live answers judged | `0.1` |
| `COST_PER_INPUT_TOKEN` / `COST_PER_OUTPUT_TOKEN` | Per-token cost tracking | `0` |

---

## Known limitations

- **Single-conversation UI** — the Streamlit app starts a new conversation per browser session; there is no conversation list/picker yet (history data is already persisted, see `conversations` / `messages`).
- **Structured outputs required for eval paths** — `generate-ground-truth` and both judge flows call `client.beta.chat.completions.parse` with a JSON schema. Your LLM endpoint must support structured outputs; many local servers (e.g. Ollama) need this enabled explicitly.
- **Doc-level relevance** — retrieval evaluation measures whether the *expected document* appears in the top-k; it does not score passage-level relevance within a document.
- **CPU baselines** — the embedder and reranker are small CPU-only ONNX models; they are fast and dependency-free, not state-of-the-art.
- **Stale eval artifacts** — the checked-in files under `data/evaluation/` come from earlier runs (some use an older schema, e.g. `avg_correctness`). Regenerate with the current commands before relying on the numbers.

---
