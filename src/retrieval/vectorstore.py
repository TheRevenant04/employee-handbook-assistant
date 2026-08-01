import os
import logging
from contextlib import contextmanager

from dotenv import load_dotenv
import psycopg
from psycopg import sql
from pgvector.psycopg import register_vector

from src.core.retry import retry_with_backoff

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
    def _do_connect():
        conn = psycopg.connect(_conninfo(), autocommit=autocommit, connect_timeout=CONNECT_TIMEOUT)
        register_vector(conn)
        return conn

    return retry_with_backoff(
        _do_connect,
        max_retries=MAX_RETRIES,
        base_delay=RETRY_BASE_DELAY,
        label="DB connect",
        logger_name=__name__,
    )


@contextmanager
def get_connection(*, autocommit=False):
    conn = connect_db(autocommit=autocommit)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_db(table_name="handbook_documents", dim=384, index_type="hnsw"):
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


def migrate_add_tsvector(table_name="handbook_documents"):
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

