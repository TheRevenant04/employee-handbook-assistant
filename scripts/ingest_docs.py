from app.ingestion.pipeline import fetch_markdown_files, init_db, embed_and_store
from app.vectorstore.pgvector_store import get_connection


if __name__ == "__main__":
    conn = get_connection()
    try:
        init_db(conn)
        documents = fetch_markdown_files()
        if documents:
            embed_and_store(conn, documents)
    finally:
        conn.close()
