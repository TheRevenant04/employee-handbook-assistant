# Setup

This guide explains how to get the Employee Handbook Assistant running. There are **two ways to set it up** — pick the one that fits how you like to work, then follow that section top to bottom.

---

## 1. Choose your setup

| | **Local (manual)** | **Docker (all-in-one)** |
| --- | --- | --- |
| What you run | The app from source on your machine, plus your own PostgreSQL | Everything in containers via `docker compose` |
| You need installed | Python 3.13+, [uv](https://docs.astral.sh/uv/), PostgreSQL with pgvector | Docker + Docker Compose |
| Database | You install/manage PostgreSQL + pgvector | Included (`pgvector/pgvector:pg18`) |
| Grafana monitoring | Not included — set up yourself | Included, pre-provisioned |
| Ingestion | Direct CLI (`ingest` command) | Direct CLI **or** Kestra workflow (included) |
| Best for | Developers who want full control and minimal overhead | A fast, reproducible, full-stack run |

**Choose Local if** you already have Python and PostgreSQL, want to inspect/tweak the code, or want to avoid Docker.
**Choose Docker if** you want the whole stack — app, vector database, Grafana dashboard, and Kestra orchestration — up with one command, and don't want to manage PostgreSQL yourself.

- Go to the **[Local setup](#3-local-setup)** → you need the *common requirements* in section 2, then section 3.
- Go to the **[Docker setup](#4-docker-setup)** → you need the *common requirements* in section 2, then section 4.

---

## 2. Common requirements (both options)

Regardless of which setup you choose, you need:

- An **OpenAI-compatible LLM endpoint** for generation. Anything that speaks the OpenAI chat-completions API works. This guide uses **Gemini** with `gemini-3.1-flash-lite`; other options include:
  - [Gemini](https://ai.google.dev/gemini-api/docs/openai) via its OpenAI-compatible endpoint (`https://generativelanguage.googleapis.com/v1beta/openai/`)
  - OpenAI
  - [Ollama](https://ollama.com/) (local, e.g. `http://localhost:11434/v1`)
  - vLLM, LM Studio, LiteLLM, etc.
- **Structured-output support** on the LLM endpoint **if** you want to use `generate-ground-truth` or `evaluate-llm`. Both call `client.beta.chat.completions.parse` with a JSON schema (`EvaluationScores` / `Questions`), which most hosted APIs support but some local servers (e.g. Ollama) only do with explicit configuration.
- **Internet access to reach the LLM endpoint** — required for every request unless your endpoint is fully local (e.g. Ollama serving on `localhost`).
- **Internet access once** — to download the two ONNX models from Hugging Face and to fetch the handbook Markdown from GitHub on the first ingestion.
- **(Optional)** a **judge LLM** (`JUDGE_*`) if you want answer-quality evaluation.
- **No GPU is required** — all inference runs on CPU.

---

## 3. Local setup

Run the app from source on your own machine, with your own PostgreSQL instance.

### 3.1 Install dependencies

```bash
# Install uv via pip (if you don't have it) — https://docs.astral.sh/uv/
pip install uv

# Clone and enter the repo
git clone https://github.com/TheRevenant04/employee-handbook-assistant.git
cd employee-handbook-assistant

# Create the virtual environment and install dependencies
uv sync
```

`uv sync` installs the runtime dependencies plus the `dev` group (pytest, etc.).

### 3.2 Configure the environment

```bash
cp .env.example .env
```

Then edit `.env` with at minimum:

```env
# LLM used to answer questions
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/   # Gemini's OpenAI-compatible endpoint
LLM_API_KEY=your-gemini-key                      # Google AI Studio API key
LLM_MODEL=gemini-3.1-flash-lite

# PostgreSQL (point at your local instance; see section 3.3)
PGDATABASE=employee_handbook
PGUSER=user
PGPASSWORD=password
PGHOST=localhost
PGPORT=5432
```

> If you're using the repo's Docker database for convenience (see 3.3, Option B), set `PGHOST=localhost` and `PGPORT=7432` instead.

### 3.3 PostgreSQL with pgvector

You need a PostgreSQL database with the `vector` extension reachable at the `PG*` variables above. Two ways to get one:

**Option A — your own PostgreSQL.** Install the `vector` extension, then create the database and schema:

```bash
uv run python scripts/init_database.py
```

`init_database.py` creates the database named by `PGDATABASE` if missing and executes `database/init.sql`. The SQL creates:

- `handbook_documents` — handbook text + `VECTOR(384)` embedding + generated `content_tsv` (with HNSW and GIN indexes)
- `conversations`, `messages` — chat persistence (with a trigger keeping `conversations.updated_at` fresh)
- `message_metrics` — latency, token usage, cost, retrieval distances
- `evaluation_runs`, `evaluation_results` — judge scores
- `error_log` — recorded exceptions

**Option B — the repo's database container.** If you have Docker, you can run just the database (no app containers) while the app runs from source:

```bash
docker compose up -d app_postgres
```

This starts `pgvector/pgvector:pg18` on host port `7432` and runs `database/init.sql` on first boot, which creates the full schema. Point `.env` at it:

```env
PGHOST=localhost
PGPORT=7432
```

> The `app_postgres` service uses a custom entrypoint (`scripts/app_db_docker_entrypoint.sh`) that ensures the target database exists and re-applies `init.sql` on every start, so schema changes are picked up automatically.

### 3.4 Download the ONNX models

```bash
uv run python scripts/download_models.py
```

Downloads from Hugging Face into `models/`:

| Model | Purpose | Path |
| --- | --- | --- |
| `Xenova/all-MiniLM-L6-v2` | Sentence embeddings (384 dims) | `models/Xenova/all-MiniLM-L6-v2/` |
| `Xenova/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker | `models/Xenova/ms-marco-MiniLM-L-6-v2/` |

`models/` is gitignored; re-run this script on a fresh checkout.

### 3.5 Run it

```bash
# Ingest the handbook (fetches .md files from GitHub, embeds, stores)
uv run python -m src.main ingest

# Start the chat UI → http://localhost:8501
uv run python -m src.main ui
```

See [`docs/usage.md`](usage.md) for the full command reference and workflow.

---

## 4. Docker setup

Run the entire stack — app, vector database, Grafana, and Kestra — in containers with one command.

### 4.1 Install Docker and clone the repo

Install **Docker** and **Docker Compose** for your OS, then clone the repo:

```bash
git clone https://github.com/TheRevenant04/employee-handbook-assistant.git
cd employee-handbook-assistant
```

### 4.2 Configure the environment

```bash
cp .env.example .env
```

Edit `.env` with your LLM settings:

```env
# LLM used to answer questions
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/   # Gemini's OpenAI-compatible endpoint
LLM_API_KEY=your-gemini-key                      # Google AI Studio API key
LLM_MODEL=gemini-3.1-flash-lite
```

**Docker-only variables.** `docker-compose.yml` references several variables that are **not** present in `.env.example` — `docker compose up` will fail without them. Add the following to `.env` when using Docker:

```env
# app_postgres bootstrap
POSTGRES_DB=employee_handbook
POSTGRES_USER=user
POSTGRES_PASSWORD=password

# Base64 of POSTGRES_PASSWORD — the Kestra flow reads it via {{ secret('PGPASSWORD') }}
SECRET_PGPASSWORD=cGFzc3dvcmQ=

# Grafana datasource (must point at the *app* PostgreSQL)
GRAFANA_PG_HOST=app_postgres:5432
GRAFANA_PG_USER=user
GRAFANA_PG_PASSWORD=password
GRAFANA_DB_NAME=employee_handbook

# Kestra orchestration — login for http://localhost:8082
KESTRA_BASIC_AUTH_USERNAME=admin
KESTRA_BASIC_AUTH_PASSWORD=changeme

# Kestra AI assistant (optional; only used by the Kestra UI)
GEMINI_API_KEY=
```

### 4.3 Start the stack

```bash
docker compose up -d
```

First run builds the app image (which downloads the ONNX models); afterwards it just starts the five services:

| Service | Image | Port | Purpose |
| --- | --- | --- | --- |
| `app` | built from `Dockerfile` | `8501` | Streamlit chat UI |
| `app_postgres` | `pgvector/pgvector:pg18` | `7432` | Vector store + app schema |
| `grafana` | `grafana/grafana:11.2.2` | `3000` | Monitoring dashboard |
| `kestra` | `kestra/kestra:v1.3.28` | `8082` | Workflow orchestration UI (runs the ingestion flow) |
| `kestra_postgres` | `postgres:18` | — | Kestra's own metadata database |

Two notes:

- **The image downloads the ONNX models at build time** (`RUN uv run python scripts/download_models.py` in the `Dockerfile`), so the first build needs network access. It does **not** run ingestion automatically.
- **Ingest once the stack is up** by running the pipeline inside the app container. Type the command in a terminal on your **host machine** — `docker compose exec app` is what runs it *inside* the container; it does not require (and won't work from) a shell already inside a container. Use plain `python`, not `uv run` — `uv` is not installed in the runtime image (only the venv, whose `python` is on `PATH`). Code is baked into the image at build time (only `/app/data` is a mounted volume; models are baked in too), so re-running ingest from the host picks up the baked-in code:

```bash
docker compose exec app python -m src.main ingest
```

### 4.4 What's running

- **Chat UI** → http://localhost:8501
- **Grafana** → http://localhost:3000 (log in with `GF_SECURITY_ADMIN_USER` / `GF_SECURITY_ADMIN_PASSWORD` from `.env`, defaults `admin` / `changeme`). It is pre-provisioned with a PostgreSQL datasource and the **Employee Handbook RAG Metrics** dashboard.
- **Kestra** → http://localhost:8082 (log in with `KESTRA_BASIC_AUTH_USERNAME` / `KESTRA_BASIC_AUTH_PASSWORD`). Use it to run the ingestion flow — see [4.5 Kestra ingestion](#45-kestra-ingestion-optional).

---

### 4.5 Kestra ingestion (optional)

Kestra is included so you can run ingestion from a UI instead of a shell. The flow and its script live in the repo:

- `kestra/flows/ingest-employee-handbook.yml` — the workflow. It downloads the Python script, then runs it in a Docker container that joins the app network (`employee-handbook-assistant_appnet`) so it can reach `app_postgres`.
- `kestra/scripts/ingest_handbook.py` — fetches the handbook from GitHub, embeds each file with ONNX Runtime (`light-embed`), and upserts it into `handbook_documents` (one row per file, no PyTorch, no API key).

1. Open Kestra at http://localhost:8082 and log in with the `KESTRA_BASIC_AUTH_*` credentials from `.env`.
2. Upload the flow from `kestra/flows/ingest-employee-handbook.yml` either way:
   - **Script (recommended)** — from the repo root, run:
     ```bash
     bash scripts/upload_kestra_flows.sh
     ```
     Git Bash/WSL/Linux/macOS. It reads `KESTRA_BASIC_AUTH_*` and `KESTRA_URL` (default `http://localhost:8082`) from your `.env`, then POSTs every `kestra/flows/*.yml` to Kestra's `/api/v1/flows/import` endpoint, which creates or updates the flow. Optional flags:
     ```bash
     bash scripts/upload_kestra_flows.sh --file kestra/flows/ingest-employee-handbook.yml
     bash scripts/upload_kestra_flows.sh --url http://localhost:8082 --username admin --password changeme
     ```
   - **Manually in the UI** — in Kestra at http://localhost:8082, open **Flows** → **Create** and paste the YAML into the editor or upload the file.
3. No database configuration needed — `docker-compose.yml` passes the app database connection into the Kestra container as `PGHOST`, `PGPORT`, `PGUSER`, `PGDATABASE` (exposed to flows as `{{ envs.pghost }}`, `{{ envs.pguser }}`, `{{ envs.pgdatabase }}`; names are lowercased under `envs.*` because `kestra.variables.env-vars-prefix` is set to `""`). The password is handled as a Kestra secret instead: set `SECRET_PGPASSWORD` (Base64 of `POSTGRES_PASSWORD`) in `.env`, and the flow resolves it via `{{ secret('PGPASSWORD') }}`. All `POSTGRES_*` values come from `.env` via compose variable interpolation.
4. Click **Execute** (optionally override inputs such as `owner` / `repo` / `branch`).

Both this flow and the direct `ingest` command store one row per handbook file, so you can switch between them freely. See [`docs/usage.md`](usage.md) for full details on the inputs.

---

## 5. Full environment variable reference

All variables are optional unless marked **required**.

### LLM (generation)

| Variable | Description | Default |
| --- | --- | --- |
| `LLM_BASE_URL` **required** | OpenAI-compatible base URL | — |
| `LLM_API_KEY` **required** | API key | — |
| `LLM_MODEL` **required** | Model name | — |

### Judge (LLM-as-judge evaluation)

| Variable | Description | Default |
| --- | --- | --- |
| `JUDGE_BASE_URL` | Judge endpoint (leave unset to disable judge) | — |
| `JUDGE_MODEL` | Judge model | — |
| `JUDGE_API_KEY` | Judge API key | — |

### PostgreSQL

| Variable | Description | Default |
| --- | --- | --- |
| `PGDATABASE` | Database name | `employee_handbook` |
| `PGUSER` | User | `user` |
| `PGPASSWORD` | Password | `password` |
| `PGHOST` | Host | `localhost` |
| `PGPORT` | Port | `5432` |

### Embeddings / ingestion

| Variable | Description | Default |
| --- | --- | --- |
| `TABLE_NAME` | Vector store table | `handbook_documents` |
| `VECTOR_DIM` | Embedding dimension | `384` |
| `MODEL_PATH` | Embedding model path | `models/Xenova/all-MiniLM-L6-v2` |
| `GITHUB_OWNER` | Handbook repo owner | `madetech` |
| `GITHUB_REPO` | Handbook repo | `handbook` |
| `GITHUB_BRANCH` | Branch | `main` |

### Reranker

| Variable | Description | Default |
| --- | --- | --- |
| `RERANKER_ENABLED` | Enable cross-encoder reranking | `false` |
| `RERANKER_MODEL_PATH` | Reranker model path | `models/Xenova/ms-marco-MiniLM-L-6-v2` |

### Query rewriter

| Variable | Description | Default |
| --- | --- | --- |
| `QUERY_REWRITER_ENABLED` | Rewrite queries before retrieval | `false` |
| `QUERY_REWRITER_MODEL` | Rewriter model (falls back to `LLM_MODEL`) | — |

### Cost tracking

| Variable | Description | Default |
| --- | --- | --- |
| `COST_PER_INPUT_TOKEN` | Cost per input token | `0` |
| `COST_PER_OUTPUT_TOKEN` | Cost per output token | `0` |
| `JUDGE_COST_PER_INPUT_TOKEN` | Judge input token cost | `0` |
| `JUDGE_COST_PER_OUTPUT_TOKEN` | Judge output token cost | `0` |

### Retrieval evaluation

| Variable | Description | Default |
| --- | --- | --- |
| `NUM_RESULTS` | Top-k for search/eval | `5` |
| `HYBRID_ALPHAS` | Comma-separated alphas to evaluate | `0.2,0.5,0.8` |
| `GROUND_TRUTH_PATH` | Ground-truth CSV | `data/ground_truth.csv` |
| `EVAL_OUTPUT_DIR` | Eval output directory | `data/evaluation` |

### Ground-truth / LLM eval

| Variable | Description | Default |
| --- | --- | --- |
| `NUM_QUESTIONS_PER_DOC` | Questions generated per document | `5` |
| `MAX_WORKERS` | Parallel workers for LLM eval | `1` |

### Rate limiting / retries

| Variable | Description | Default |
| --- | --- | --- |
| `MAX_EVAL_RPM` | Judge requests per minute (online eval) | `10` |
| `MAX_JUDGE_RPM` | Judge requests per minute (batch eval) | `10` |
| `MAX_REQUESTS_PER_MINUTE` | Ground-truth gen RPM | `10` |
| `MAX_RETRIES` | Retries for LLM/DB calls | `5` |
| `RETRY_BASE_DELAY` | Initial backoff delay (seconds) | `2` |
| `EVAL_SAMPLE_RATE` | Fraction of live answers judged | `0.1` |

### Grafana / Docker compose

| Variable | Description | Default |
| --- | --- | --- |
| `GF_SECURITY_ADMIN_USER` | Grafana admin user | `admin` |
| `GF_SECURITY_ADMIN_PASSWORD` | Grafana admin password | `changeme` |
| `GRAFANA_PG_HOST` / `GRAFANA_PG_USER` / `GRAFANA_PG_PASSWORD` / `GRAFANA_DB_NAME` | Datasource connection for the dashboard | — |
| `POSTGRES_DB` / `POSTGRES_USER` / `POSTGRES_PASSWORD` | Docker `app_postgres` bootstrap | — |
| `SECRET_PGPASSWORD` | Base64 of `POSTGRES_PASSWORD`; the Kestra flow reads it via `{{ secret('PGPASSWORD') }}` | — |
| `KESTRA_BASIC_AUTH_USERNAME` / `KESTRA_BASIC_AUTH_PASSWORD` | Kestra UI login | `admin` / `changeme` |
| `GEMINI_API_KEY` | API key for the Kestra AI assistant (optional) | — |

### Advanced (read in code, not in `.env.example`)

These are honoured by the code but aren't shipped in the example env file:

| Variable | Description | Default |
| --- | --- | --- |
| `CONNECT_TIMEOUT` | DB connect timeout (seconds) | `10` |
| `LOG_LEVEL` | Root log level (`DEBUG`/`INFO`/`WARNING`/`ERROR`) | `INFO` |
| `LOG_FORMAT` | `text` (human-readable) or `json` (structured) | `text` |
| `RAG_MAX_RETRIES` | Retries for a single LLM answer call | `3` |
| `RAG_RETRY_BASE_DELAY` | Backoff base for the LLM answer call (seconds) | `2` |
| `EVAL_WORKERS` | Threads for the online sampled judge | `1` |
| `INIT_SQL_PATH` | Schema file for `scripts/init_database.py` | `database/init.sql` |

---

## 6. Troubleshooting

**"Model not found at models/..."**
Run `uv run python scripts/download_models.py` (see section 3.4 for the local path; models are downloaded automatically during the Docker build).

**`connection refused` on `PGPORT`**
The local path defaults to `5432`. If you're using the repo's Docker database, set `PGPORT=7432`.

**Judge evaluation never runs**
`Evaluator` is disabled unless `JUDGE_BASE_URL`, `JUDGE_MODEL`, and `JUDGE_API_KEY` are all set. On start you'll see a warning: *"Evaluator disabled: set JUDGE_BASE_URL, JUDGE_MODEL, JUDGE_API_KEY"*.

**Keyword / hybrid search returns no results**
The `content_tsv` column and its GIN index must exist. Run `scripts/init_database.py` (or restart `app_postgres`, which re-applies `init.sql`).

**Kestra flow fails with `could not translate host name "app_postgres" to address`**
The task container isn't joining the app network. Make sure the flow's `taskRunner` sets `networkMode: employee-handbook-assistant_appnet` (see `kestra/flows/ingest-employee-handbook.yml`) and that the stack was recreated with `docker compose up -d` after the network was added.

**Kestra flow fails with `ON CONFLICT DO UPDATE command cannot affect row a second time`**
The table stores one row per handbook file (`path` is unique). Use the checked-in `kestra/scripts/ingest_handbook.py`, which embeds whole files — don't chunk into multiple rows per file.

**Changing the vector dimension**
`VECTOR_DIM` must match the embedding model (384 for `all-MiniLM-L6-v2`). Changing it requires recreating the `handbook_documents` table and re-ingesting.

**LLM calls fail repeatedly**
Ensure `LLM_BASE_URL` is reachable and the model name matches what the endpoint serves. Transient 429/5xx/timeouts are retried automatically with exponential backoff.
