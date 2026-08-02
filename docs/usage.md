# Usage

This guide explains every command, walks through a realistic workflow, shows example inputs and outputs, and describes how to capture screenshots and a demo video for the README.

---

## 1. Command reference

All commands go through the single CLI entry point [`src/main.py`](../src/main.py):

```bash
uv run python -m src.main <command>
```

If you set the project up **with Docker**, run the same commands from your **host terminal**, prefixed with `docker compose exec app` — but use plain `python`, not `uv run`:

```bash
docker compose exec app python -m src.main <command>
```

`docker compose exec` is how you run a command *inside* the app container; it is typed on the host. Two things to note:

- **`uv` is not installed in the container.** The runtime image is plain Python with `/app/.venv/bin` on `PATH`, so the container entry point is `python -m src.main <command>` (that `python` is the venv's). Don't copy the local setup's `uv run` prefix — it will fail with `exec: "uv": executable file not found in $PATH`.
- Do **not** type the `docker compose exec` form from a shell that is already inside the container — `docker` is only installed on the host, so you'd get `docker: not found`. If you're already inside the container (e.g. from `docker compose exec app sh`), run the command without the prefix: `python -m src.main <command>`, then `exit` to get back to the host.

| Command | What it does | Output |
| --- | --- | --- |
| `ui` | Launch the Streamlit chat UI | http://localhost:8501 |
| `ingest` | Fetch `.md` files from GitHub, embed them, store in pgvector | Rows in `handbook_documents` |
| `generate-ground-truth` | Ask the LLM to write Q&A pairs per document | `data/ground_truth.csv` |
| `evaluate-search` | Score every retrieval method (Hit Rate, MRR) | `data/evaluation/evaluation_summary.csv` / `.json`, `evaluation_debug.csv` |
| `evaluate-llm` | LLM-as-judge scoring of generated answers | `data/evaluation/llm_evaluation_detail.csv`, `llm_evaluation_summary.json` |

### Helper scripts

| Command | What it does |
| --- | --- |
| `uv run python scripts/download_models.py` | Download the ONNX embedder + reranker from Hugging Face |
| `uv run python scripts/init_database.py` | Create the database and apply `database/init.sql` |

---

## 2. Typical workflow

Run whichever path matches the setup you chose in the [Setup guide](setup.md).

### Local setup (running from source)

```bash
# 0. Prereqs (once per checkout)
pip install uv
git clone https://github.com/TheRevenant04/employee-handbook-assistant.git
cd employee-handbook-assistant
uv sync
cp .env.example .env            # edit it!
uv run python scripts/download_models.py

# 1. Database
uv run python scripts/init_database.py          # or: docker compose up -d app_postgres

# 2. Fill the vector store
uv run python -m src.main ingest

# 3. (Optional) Build a ground-truth dataset for evaluation
uv run python -m src.main generate-ground-truth

# 4. (Optional) Compare retrieval strategies
uv run python -m src.main evaluate-search

# 5. (Optional) Judge answer quality end-to-end (requires JUDGE_* vars)
uv run python -m src.main evaluate-llm

# 6. Chat with the assistant
uv run python -m src.main ui
```

### Docker setup (everything in containers)

> All commands below are typed in a terminal on your **host machine**. `docker compose exec app` runs a command *inside* the app container — you don't need (and mustn't already be in) a container shell to use it. If you're already inside a container, run `exit` first, then run these from the host.

```bash
# 0. Start the stack in the background (this also starts the UI)
#    First run builds the image; afterwards it just starts the containers.
docker compose up -d

# 1. Database
#    Already provisioned — the app_postgres container applies database/init.sql on boot.

# 2. Fill the vector store (runs inside the app container, typed from the host)
docker compose exec app python -m src.main ingest

# 3. (Optional) Build a ground-truth dataset for evaluation
docker compose exec app python -m src.main generate-ground-truth

# 4. (Optional) Compare retrieval strategies
docker compose exec app python -m src.main evaluate-search

# 5. (Optional) Judge answer quality end-to-end (requires JUDGE_* vars)
docker compose exec app python -m src.main evaluate-llm

# 6. Chat with the assistant → http://localhost:8501
#    (the UI is already up from step 0)
```

### A note on order

- **Ingest must run before the UI or any evaluation** — search queries the `handbook_documents` table.
- **`generate-ground-truth` needs the ingest step** to know which documents exist (it reads them from the database).
- **`evaluate-search` needs `data/ground_truth.csv`** — use the checked-in file or generate your own.

---

## 3. The chat UI (`ui`)

Open http://localhost:8501.

- Type a question in the chat input, e.g. **"How much annual leave do I get?"**
- The assistant retrieves relevant handbook sections and answers **only from them**.
- Every answer gets 👍 / 👎 buttons. Ratings are stored on the message and can be correlated with automatic judge scores.
- Conversations, messages, latency, token counts, and costs are persisted to PostgreSQL (`conversations`, `messages`, `message_metrics`). The current UI starts a fresh conversation per browser session; the persisted history is what a future conversation-picker feature will read from.

### Example conversation

| You | Assistant |
| --- | --- |
| *How many days of paid annual leave am I entitled to take?* | Made Tech offers **30 days' paid annual leave** each year, as stated in the Benefits section of the handbook (see `benefits/taking_holiday.md`). |
| *Can I use the Cycle to Work scheme just for accessories?* | Yes. The handbook states you are welcome to use the Cycle to Work scheme for accessories only (e.g. helmet, clothing, locks, lights) — this is a popular option for employees who already own a bike. |
| *How do I get reimbursed for travel to a client site?* | The handbook covers this under **Expenses** (`guides/compensation/expenses.md`). If you're a client-facing team member, you can expense travel to client sites outside your regional office through the expenses policy. |
| *What happens if I'm diagnosed with a rare tropical disease?* | I don't know based on the information available in the handbook. |

> The last answer is the designed **safe fallback**: when the handbook is silent, the assistant says so instead of inventing a policy. You can reproduce this behaviour by asking an out-of-scope question.

### What each answer records

For every message, the system stores (in `message_metrics`):

- Total / retrieval / LLM latency (ms)
- Number of results and the average & minimum embedding distance
- Input & output tokens and estimated cost
- The model used and success/failure

---

## 4. Ingestion (two options)

There are **two ways to fill the vector store** — pick whichever fits how you work:

| Option | When to use it |
| --- | --- |
| **Direct CLI** (`ingest`) | Local setup, or a quick one-off run from the Docker stack |
| **Kestra flow** | Docker setup — run and repeat ingestion from a UI, no shell access needed |

Both approaches store **one row per handbook file** (upserted on `path`), so they are interchangeable and re-running either one is safe.

### Option 1 — Direct (`ingest`)

Reads `GITHUB_OWNER/GITHUB_REPO/GITHUB_BRANCH`, lists every `.md` file in the repo tree, downloads each, embeds the content with `all-MiniLM-L6-v2`, and inserts rows into `TABLE_NAME` (upsert on `path`).

```bash
uv run python -m src.main ingest
```

Example log tail:

```
[INFO] Fetching file tree from https://api.github.com/repos/madetech/handbook/git/trees/main?recursive=1
[INFO] Found 123 markdown files
[INFO] Embedding 123 documents...
[INFO] Stored 123 documents in PostgreSQL
[INFO] === Ingest complete ===
```

Use a different handbook without touching code:

```bash
# Windows PowerShell example
$env:GITHUB_OWNER="your-org"; $env:GITHUB_REPO="your-handbook"; $env:GITHUB_BRANCH="main"
uv run python -m src.main ingest
```

### Option 2 — Kestra flow

If you run the **Docker** stack, Kestra is included (see [`docs/setup.md`](setup.md)). It runs the same ingestion from a workflow UI, so you never need a terminal to ingest.

- **UI:** http://localhost:8082 (login with `KESTRA_BASIC_AUTH_USERNAME` / `KESTRA_BASIC_AUTH_PASSWORD` from `.env`)
- **Flow source:** [`kestra/flows/ingest-employee-handbook.yml`](../kestra/flows/ingest-employee-handbook.yml)
- **Script:** [`kestra/scripts/ingest_handbook.py`](../kestra/scripts/ingest_handbook.py) — downloaded at runtime from its raw GitHub URL, so the flow always runs the version committed to the repo

**How it works.** The flow has two tasks:

1. `download_script` — downloads `ingest_handbook.py` from `inputs.scriptUrl`.
2. `process_and_embed` — runs the script in a Docker container that joins the app network (`employee-handbook-assistant_appnet`), fetches the handbook from GitHub, embeds each file with ONNX Runtime (`light-embed`), and upserts it into `handbook_documents`.

**Run it.** In the Kestra UI, create a flow from the YAML above (paste it into the editor or upload the file), then click **Execute**. The flow takes these inputs:

| Input | Default | Description |
| --- | --- | --- |
| `owner` / `repo` / `branch` | `madetech` / `handbook` / `main` | Handbook source |
| `embeddingModel` | `onnx-models/all-MiniLM-L6-v2-onnx` | ONNX embedder (no PyTorch, no API key) |
| `modelCacheDir` | `/tmp/kestra` | Directory to cache the embedding model |
| `scriptUrl` | raw GitHub URL of `kestra/scripts/ingest_handbook.py` | Script to download and run |

**Database connection.** The flow reads the app database connection from `docker-compose.yml`, which passes `PGHOST`, `PGPORT`, `PGUSER`, `PGDATABASE` into the Kestra container (exposed to flows as `{{ envs.pghost }}`, `{{ envs.pguser }}`, `{{ envs.pgdatabase }}` via `kestra.variables.env-vars-prefix: ""`, which strips no prefix and lowercases names). The password uses Kestra's secret mechanism instead: `{{ secret('PGPASSWORD') }}` resolves the Base64-encoded `SECRET_PGPASSWORD` env var from `.env`, keeping it out of the flow definition and masking it in Kestra logs. No setup in the Kestra **Environments** page is needed.

---

## 5. Ground-truth generation (`generate-ground-truth`)

Reads every document from the database and asks `LLM_MODEL` to produce `NUM_QUESTIONS_PER_DOC` natural questions an employee might ask, each mapped to the document that answers it.

```bash
uv run python -m src.main generate-ground-truth
```

Resulting `data/ground_truth.csv`:

```csv
question,document
How many days of paid annual leave am I entitled to take?,README.md
Is there a budget available to help me purchase extra equipment for my home office?,README.md
What is the maximum amount I can spend on a new bike through the scheme?,benefits/cycle_to_work_scheme.md
Can I use the cycle scheme if I only want to pick up some new accessories instead of a whole bike?,benefits/cycle_to_work_scheme.md
```

---

## 6. Retrieval evaluation (`evaluate-search`)

Scores **Hit Rate @ k** and **MRR** for each method over the whole ground-truth set.

```bash
uv run python -m src.main evaluate-search
```

Output (console + `data/evaluation/`):

```
RETRIEVAL EVALUATION RESULTS
============================================================
method              hit_rate   mrr   questions_evaluated
rerank_hybrid_0.5   0.521      0.411  925
rerank_hybrid_0.8   0.521      0.411  925
rerank_hybrid_0.2   0.521      0.411  925
hybrid_0.5          0.483      0.357  925
...
Best method: rerank_hybrid_0.5 (MRR=0.411, Hit Rate=0.521)
```

Files written:

- `evaluation_summary.csv` — one row per method, sorted by MRR
- `evaluation_debug.csv` — per-question hit/rank detail
- `evaluation_summary.json` — same as CSV, machine-readable

---

## 7. LLM-as-judge evaluation (`evaluate-llm`)

Runs the full RAG pipeline on every ground-truth question, then has `JUDGE_MODEL` score the answer on **Faithfulness**, **Context relevance**, and **Completeness** (1–5).

Requires `JUDGE_BASE_URL`, `JUDGE_MODEL`, and `JUDGE_API_KEY`. The judge endpoint must support **structured outputs** (the call uses `response_format` with the `EvaluationScores` schema).

```bash
uv run python -m src.main evaluate-llm
```

Example output (representative — regenerate with your own judge/model):

```
LLM EVALUATION RESULTS
============================================================
Questions evaluated: 10
Avg Faithfulness:       2.80 / 5
Avg Context Relevance:  3.20 / 5
Avg Completeness:       2.90 / 5
Faithfulness dist:      {'1': 3, '2': 2, '3': 2, '4': 0, '5': 3}
```

Details (scores + judge reasoning per question) go to `data/evaluation/llm_evaluation_detail.csv`; aggregates to `llm_evaluation_summary.json`. Results are also persisted to the `evaluation_runs` / `evaluation_results` tables.

> **Heads-up:** the `llm_evaluation_summary.json` checked into the repo was produced by an earlier version and uses the old field names (`avg_correctness`, `correctness_distribution`). The current code emits `context_relevance` instead — re-run the command to get output matching this guide.

---

## 8. Online (sampled) evaluation

When the UI runs with `JUDGE_*` set, a background worker samples a fraction (`EVAL_SAMPLE_RATE`, default 10%) of live answers and judges them asynchronously — no manual trigger needed. Judge scores land in `evaluation_results` and can be joined to user ratings via the `messages` table.

---

## 9. Monitoring with Grafana

If running via `docker compose`, open http://localhost:3000 and sign in (default `admin` / `changeme`).

The provisioned **Employee Handbook RAG Metrics** dashboard queries the app PostgreSQL instance directly and includes panels for:

- Total messages / conversations
- Latency over time (total, retrieval, LLM)
- Token usage and estimated cost
- Retrieval distance (avg / min)
- Success rate and error counts (from `error_log`)
- Evaluation scores (faithfulness / context relevance / completeness)

Dashboards are defined in `grafana/dashboards/employee-handbook-metrics.json` and auto-loaded by the provisioning config in `grafana/provisioning/`.
