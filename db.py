import os
import time
import logging
from contextlib import contextmanager

from dotenv import load_dotenv
import psycopg
from psycopg import sql
from pgvector.psycopg import register_vector

logger = logging.getLogger(__name__)
load_dotenv()

MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "5"))
RETRY_BASE_DELAY: float = float(os.getenv("RETRY_BASE_DELAY", "2"))
CONNECT_TIMEOUT: int = int(os.getenv("CONNECT_TIMEOUT", "10"))


def _conninfo():
    return psycopg.conninfo.make_conninfo(
        dbname=os.getenv("PGDATABASE", "employee_handbook"),
        user=os.getenv("PGUSER", "user"),
        password=os.getenv("PGPASSWORD", "password"),
        host=os.getenv("PGHOST", "localhost"),
        port=os.getenv("PGPORT", "5432"),
    )


def connect_db(*, autocommit=False):
    for attempt in range(MAX_RETRIES):
        try:
            conn = psycopg.connect(_conninfo(), autocommit=autocommit, connect_timeout=CONNECT_TIMEOUT)
            register_vector(conn)
            return conn
        except Exception as e:
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            if attempt < MAX_RETRIES - 1:
                logger.warning(
                    "DB connect failed (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt + 1, MAX_RETRIES, e, delay,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "DB connect failed after %d attempts: %s",
                    MAX_RETRIES, e,
                    exc_info=True,
                )
                raise


@contextmanager
def get_connection(*, autocommit=False):
    conn = connect_db(autocommit=autocommit)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_db(table_name="employee_handbook", dim=384, index_type="hnsw"):
    if not isinstance(dim, int) or dim <= 0:
        raise ValueError("dim must be a positive integer")

    index_type = index_type.lower()
    if index_type not in {"hnsw", "ivfflat"}:
        raise ValueError("index_type must be 'hnsw' or 'ivfflat'")

    with get_connection(autocommit=True) as conn:
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
                            to_tsvector('english', coalesce(content, ''))
                        ) STORED
                    );
                    """
                ).format(
                    table=sql.Identifier(table_name),
                    dim=sql.SQL(str(dim)),
                )
            )

            if index_type == "ivfflat":
                cur.execute(
                    sql.SQL(
                        """
                        CREATE INDEX IF NOT EXISTS {index_name}
                        ON {table}
                        USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100);
                        """
                    ).format(
                        index_name=sql.Identifier(f"{table_name}_embedding_ivfflat_idx"),
                        table=sql.Identifier(table_name),
                    )
                )
            else:
                cur.execute(
                    sql.SQL(
                        """
                        CREATE INDEX IF NOT EXISTS {index_name}
                        ON {table}
                        USING hnsw (embedding vector_cosine_ops);
                        """
                    ).format(
                        index_name=sql.Identifier(f"{table_name}_embedding_hnsw_idx"),
                        table=sql.Identifier(table_name),
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
                    index_name=sql.Identifier(f"{table_name}_content_tsv_idx"),
                    table=sql.Identifier(table_name),
                )
            )

    logger.info("Database initialized for table %s", table_name)


def migrate_add_tsvector(table_name="employee_handbook"):
    with get_connection(autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    ALTER TABLE {table}
                    ADD COLUMN IF NOT EXISTS content_tsv TSVECTOR
                    GENERATED ALWAYS AS (to_tsvector('english', coalesce(content, ''))) STORED;
                    """
                ).format(table=sql.Identifier(table_name))
            )

            cur.execute(
                sql.SQL(
                    """
                    CREATE INDEX IF NOT EXISTS {index_name}
                    ON {table}
                    USING GIN (content_tsv);
                    """
                ).format(
                    index_name=sql.Identifier(f"{table_name}_content_tsv_idx"),
                    table=sql.Identifier(table_name),
                )
            )

    logger.info("Migration complete: added content_tsv column and GIN index")


def init_llm_evaluation_schema():
    with get_connection(autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("CREATE SCHEMA IF NOT EXISTS {schema}").format(
                    schema=sql.Identifier("llm_evaluation")
                )
            )

            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {schema}.{table} (
                        id BIGSERIAL PRIMARY KEY,
                        timestamp TIMESTAMPTZ NOT NULL DEFAULT now(),
                        judge_model TEXT NOT NULL,
                        evaluated_model TEXT NOT NULL,
                        num_questions INT NOT NULL CHECK (num_questions >= 0),
                        config JSONB
                    );
                    """
                ).format(
                    schema=sql.Identifier("llm_evaluation"),
                    table=sql.Identifier("evaluation_runs"),
                )
            )

            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {schema}.{table} (
                        id BIGSERIAL PRIMARY KEY,
                        run_id BIGINT NOT NULL REFERENCES {schema}.{runs_table}(id) ON DELETE CASCADE,
                        message_id BIGINT NOT NULL REFERENCES chat.messages(id),
                        expected_document TEXT,
                        retrieved_context TEXT,
                        faithfulness_score SMALLINT NOT NULL CHECK (faithfulness_score BETWEEN 1 AND 5),
                        faithfulness_reasoning TEXT,
                        context_relevance_score SMALLINT NOT NULL CHECK (context_relevance_score BETWEEN 1 AND 5),
                        context_relevance_reasoning TEXT,
                        completeness_score SMALLINT NOT NULL CHECK (completeness_score BETWEEN 1 AND 5),
                        completeness_reasoning TEXT,
                        judge_input_tokens INT,
                        judge_output_tokens INT,
                        judge_cost NUMERIC(12, 6) NOT NULL DEFAULT 0
                    );
                    """
                ).format(
                    schema=sql.Identifier("llm_evaluation"),
                    table=sql.Identifier("evaluation_results"),
                    runs_table=sql.Identifier("evaluation_runs"),
                )
            )

            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {idx} ON {schema}.{table} (run_id)"
                ).format(
                    idx=sql.Identifier("idx_eval_results_run"),
                    schema=sql.Identifier("llm_evaluation"),
                    table=sql.Identifier("evaluation_results"),
                )
            )

            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {idx} ON {schema}.{table} (message_id)"
                ).format(
                    idx=sql.Identifier("idx_eval_results_msg"),
                    schema=sql.Identifier("llm_evaluation"),
                    table=sql.Identifier("evaluation_results"),
                )
            )

    logger.info("LLM evaluation schema initialized")