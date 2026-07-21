import os
import psycopg
from psycopg import sql
from pgvector.psycopg import register_vector


def connect_db():
    return psycopg.connect(
        dbname=os.getenv("PGDATABASE", "employee_handbook"),
        user=os.getenv("PGUSER", "user"),
        password=os.getenv("PGPASSWORD", "password"),
        host=os.getenv("PGHOST", "localhost"),
        port=int(os.getenv("PGPORT", "5432")),
    )

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
                        embedding VECTOR({dim}) NOT NULL
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

        conn.commit()
    finally:
        conn.close()


def get_db_connection():
    conn = connect_db()
    register_vector(conn)
    return conn