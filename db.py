import os
import time
import logging
from dotenv import load_dotenv
import psycopg
from psycopg import sql
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector

logger = logging.getLogger(__name__)

load_dotenv()

_pool = None

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
RETRY_BASE_DELAY = float(os.getenv("RETRY_BASE_DELAY", "2"))


def _conninfo():
    return psycopg.conninfo.make_conninfo(
        dbname=os.getenv("PGDATABASE", "employee_handbook"),
        user=os.getenv("PGUSER", "user"),
        password=os.getenv("PGPASSWORD", "password"),
        host=os.getenv("PGHOST", "localhost"),
        port=os.getenv("PGPORT", "5432"),
    )


def _configure_conn(conn):
    register_vector(conn)


def _get_pool():
    global _pool
    if _pool is None:
        for attempt in range(MAX_RETRIES):
            try:
                _pool = ConnectionPool(
                    conninfo=_conninfo(),
                    min_size=1,
                    max_size=10,
                    configure=_configure_conn,
                    check=sql.SQL("SELECT 1"),
                )
                logger.info("Connection pool initialized (min=1, max=10)")
                return _pool
            except Exception as e:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                if attempt < MAX_RETRIES - 1:
                    logger.warning(
                        "Pool init failed (attempt %d/%d): %s. Retrying in %.1fs...",
                        attempt + 1, MAX_RETRIES, e, delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error("Pool init failed after %d attempts: %s", MAX_RETRIES, e)
                    raise
    return _pool


def get_connection():
    pool = _get_pool()
    return pool.connection()


def connect_db():
    for attempt in range(MAX_RETRIES):
        try:
            return psycopg.connect(
                dbname=os.getenv("PGDATABASE", "employee_handbook"),
                user=os.getenv("PGUSER", "user"),
                password=os.getenv("PGPASSWORD", "password"),
                host=os.getenv("PGHOST", "localhost"),
                port=int(os.getenv("PGPORT", "5432")),
            )
        except Exception as e:
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            if attempt < MAX_RETRIES - 1:
                logger.warning(
                    "DB connect failed (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt + 1, MAX_RETRIES, e, delay,
                )
                time.sleep(delay)
            else:
                logger.error("DB connect failed after %d attempts: %s", MAX_RETRIES, e)
                raise

def init_db(
    table_name="employee_handbook",
    dim=384,
    index_type="hnsw",
):
    if not isinstance(dim, int) or dim <= 0:
        raise ValueError("dim must be a positive integer")

    conn = connect_db()
    try:
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
                    table=sql.Identifier(table_name),
                    dim=sql.SQL(str(dim)),
                )
            )

            if index_type.lower() == "ivfflat":
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

        conn.commit()
    finally:
        conn.close()


def migrate_add_tsvector(table_name="employee_handbook"):
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    ALTER TABLE {table}
                    ADD COLUMN IF NOT EXISTS content_tsv TSVECTOR
                    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;
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

        conn.commit()
        logger.info("Migration complete: added content_tsv column and GIN index")
    finally:
        conn.close()


def get_db_connection():
    conn = connect_db()
    register_vector(conn)
    return conn


def init_llm_evaluation_schema():
    conn = connect_db()
    try:
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
                        id              BIGSERIAL PRIMARY KEY,
                        timestamp       TIMESTAMPTZ NOT NULL DEFAULT now(),
                        judge_model     TEXT NOT NULL,
                        evaluated_model TEXT NOT NULL,
                        num_questions   INT NOT NULL,
                        config          JSONB
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
                        id                         BIGSERIAL PRIMARY KEY,
                        run_id                     BIGINT NOT NULL REFERENCES {schema}.{runs_table}(id),
                        message_id                 BIGINT NOT NULL REFERENCES chat.messages(id),
                        expected_document          TEXT,
                        retrieved_context          TEXT,
                        faithfulness_score         SMALLINT NOT NULL CHECK (faithfulness_score BETWEEN 1 AND 5),
                        faithfulness_reasoning     TEXT,
                        context_relevance_score    SMALLINT NOT NULL CHECK (context_relevance_score BETWEEN 1 AND 5),
                        context_relevance_reasoning TEXT,
                        completeness_score         SMALLINT NOT NULL CHECK (completeness_score BETWEEN 1 AND 5),
                        completeness_reasoning     TEXT,
                        judge_input_tokens         INT,
                        judge_output_tokens        INT,
                        judge_cost                 FLOAT DEFAULT 0
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

        conn.commit()
        logger.info("LLM evaluation schema initialized")
    finally:
        conn.close()