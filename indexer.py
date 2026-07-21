from psycopg import sql
from embedder import Embedder
from db import init_db


class PostgresVectorIndex:
    def __init__(
        self,
        conn,
        table_name="employee_handbook",
        model_path="models/Xenova/all-MiniLM-L6-v2",
    ):
        self.conn = conn
        self.table_name = table_name
        self.embedder = Embedder(model_path)

    def exists(self):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name = %s
                )
                """,
                (self.table_name,),
            )
            return cur.fetchone()[0]

    def has_documents(self):
        if not self.exists():
            return False

        with self.conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT EXISTS (SELECT 1 FROM {table} LIMIT 1)").format(
                    table=sql.Identifier(self.table_name)
                )
            )
            return cur.fetchone()[0]

    def build_if_missing(self, documents, dim=384, index_type="hnsw"):
        init_db(
            table_name=self.table_name,
            dim=dim,
            index_type=index_type,
        )

        if self.has_documents():
            return self

        self.add(documents)
        return self

    def add(self, documents):
        if not documents:
            return 0

        contents = [d["content"] for d in documents]
        vectors = self.embedder.encode_batch(contents, normalize=True)

        rows = [
            (doc["path"], doc["content"], vec)
            for doc, vec in zip(documents, vectors)
        ]

        with self.conn.cursor() as cur:
            insert_sql = sql.SQL(
                """
                INSERT INTO {table} (path, content, embedding)
                VALUES (%s, %s, %s)
                ON CONFLICT (path) DO NOTHING
                """
            ).format(table=sql.Identifier(self.table_name))

            cur.executemany(insert_sql, rows)

        self.conn.commit()
        return len(rows)
