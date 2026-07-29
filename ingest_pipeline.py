import os
import sys
import logging
import requests
from pathlib import Path
from psycopg import sql

from db import get_connection
from logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

GITHUB_OWNER: str = os.getenv("GITHUB_OWNER", "madetech")
GITHUB_REPO: str = os.getenv("GITHUB_REPO", "handbook")
GITHUB_BRANCH: str = os.getenv("GITHUB_BRANCH", "main")

TABLE_NAME: str = os.getenv("TABLE_NAME", "employee_handbook")
VECTOR_DIM: int = int(os.getenv("VECTOR_DIM", "384"))
MODEL_PATH: str = os.getenv("MODEL_PATH", "models/Xenova/all-MiniLM-L6-v2")


def init_db(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {table} (
                    id BIGSERIAL PRIMARY KEY,
                    path TEXT UNIQUE NOT NULL,
                    content TEXT NOT NULL,
                    embedding VECTOR({dim}) NOT NULL,
                    content_tsv TSVECTOR GENERATED ALWAYS AS (
                        to_tsvector('english', content)
                    ) STORED
                );
                """
            ).format(
                table=sql.Identifier(TABLE_NAME),
                dim=sql.SQL(str(VECTOR_DIM)),
            )
        )

        cur.execute(
            sql.SQL(
                """
                CREATE INDEX IF NOT EXISTS {index_name}
                ON {table}
                USING hnsw (embedding vector_cosine_ops);
                """
            ).format(
                index_name=sql.Identifier(f"{TABLE_NAME}_embedding_hnsw_idx"),
                table=sql.Identifier(TABLE_NAME),
            )
        )

        cur.execute(
            sql.SQL(
                """
                CREATE INDEX IF NOT EXISTS {index_name}
                ON {table}
                USING GIN (content_tsv);
                """
            ).format(
                index_name=sql.Identifier(f"{TABLE_NAME}_content_tsv_idx"),
                table=sql.Identifier(TABLE_NAME),
            )
        )

    conn.commit()
    logger.info("Database initialized with table '%s'", TABLE_NAME)


def fetch_markdown_files() -> list[dict[str, str]]:
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/vnd.github+json",
            "User-Agent": "kestra-ingest-worker",
        }
    )

    tree_url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/git/trees/{GITHUB_BRANCH}?recursive=1"
    logger.info("Fetching file tree from %s", tree_url)
    tree_response = session.get(tree_url, timeout=30)
    tree_response.raise_for_status()
    tree_data = tree_response.json()

    markdown_paths = [
        item["path"]
        for item in tree_data.get("tree", [])
        if item.get("type") == "blob" and item.get("path", "").endswith(".md")
    ]
    logger.info("Found %d markdown files", len(markdown_paths))

    files: list[dict[str, str]] = []
    for i, path in enumerate(markdown_paths, 1):
        raw_url = f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{path}"
        response = session.get(raw_url, timeout=30)
        response.raise_for_status()
        files.append({"path": path, "content": response.text})

        if i % 10 == 0:
            logger.info("Fetched %d/%d files", i, len(markdown_paths))

    logger.info("Fetched all %d markdown files", len(files))
    return files


def embed_and_store(conn, documents: list[dict[str, str]]):
    from embedder import Embedder

    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        logger.error("Model not found at %s. Run 'uv run python download.py' first.", MODEL_PATH)
        sys.exit(1)

    logger.info("Loading embedding model from %s", MODEL_PATH)
    embedder = Embedder(MODEL_PATH)

    contents = [d["content"] for d in documents]
    logger.info("Embedding %d documents...", len(contents))
    vectors = embedder.encode_batch(contents, normalize=True)

    rows = [
        (doc["path"], doc["content"], vec)
        for doc, vec in zip(documents, vectors)
    ]

    with conn.cursor() as cur:
        insert_sql = sql.SQL(
            """
            INSERT INTO {table} (path, content, embedding)
            VALUES (%s, %s, %s)
            ON CONFLICT (path) DO NOTHING
            """
        ).format(table=sql.Identifier(TABLE_NAME))

        cur.executemany(insert_sql, rows)

    conn.commit()
    logger.info("Stored %d documents in PostgreSQL", len(rows))


def main():
    logger.info("=== Employee Handbook Ingest Pipeline ===")
    logger.info("GitHub: %s/%s @ %s", GITHUB_OWNER, GITHUB_REPO, GITHUB_BRANCH)

    with get_connection() as conn:
        init_db(conn)

        documents = fetch_markdown_files()
        if not documents:
            logger.warning("No documents found. Exiting.")
            return

        embed_and_store(conn, documents)
        logger.info("=== Ingest complete ===")


if __name__ == "__main__":
    main()