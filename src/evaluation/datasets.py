import os
import json
import logging
import pandas as pd
from openai import OpenAI
from psycopg import sql
from tqdm.auto import tqdm

from src.domain.dataset import Questions
from src.retrieval.vectorstore import get_connection
from src.utils.logging import configure_logging
from src.utils.retry import RateLimiter, is_transient_llm_error, retry_with_backoff

configure_logging()
logger = logging.getLogger(__name__)

TABLE_NAME: str = os.getenv("TABLE_NAME", "handbook_documents")
NUM_QUESTIONS_PER_DOC: int = int(os.getenv("NUM_QUESTIONS_PER_DOC", "5"))
MAX_REQUESTS_PER_MINUTE: int = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "10"))
MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "5"))
RETRY_BASE_DELAY: float = float(os.getenv("RETRY_BASE_DELAY", "10"))


DATA_GEN_INSTRUCTIONS = """
You emulate an employee who's reading the company handbook.
Formulate 5 questions this employee might ask based on a handbook document.
The document should contain the answer to the questions, and the questions should be complete and not too short.
If possible, use as fewer words as possible from the document.
The output should resemble how people ask questions on the internet.
Not too formal, not too short, not too long.
""".strip()


def load_documents():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "SELECT path, content FROM {table}"
                ).format(table=sql.Identifier(TABLE_NAME))
            )
            rows = cur.fetchall()
        return [{"path": row[0], "content": row[1]} for row in rows]
    finally:
        conn.close()


def generate_questions_for_doc(client, doc):
    model = os.getenv("LLM_MODEL")

    user_prompt = json.dumps({"path": doc["path"], "content": doc["content"]})

    messages = [
        {"role": "developer", "content": DATA_GEN_INSTRUCTIONS},
        {"role": "user", "content": user_prompt},
    ]

    def _call() -> list[dict[str, str]]:
        response = client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=Questions,
        )

        result = response.choices[0].message.parsed
        if result is None:
            logger.warning("Failed to parse response for doc: %s", doc["path"])
            return []

        records = []
        for q in result.questions:
            records.append({
                "question": q,
                "document": doc["path"],
            })
        return records

    return retry_with_backoff(
        _call,
        max_retries=MAX_RETRIES,
        base_delay=RETRY_BASE_DELAY,
        is_retryable=is_transient_llm_error,
        label=f"generate_questions({doc['path']})",
        logger_name=__name__,
    )


def main():
    logger.info("=== Ground Truth Generation ===")

    model = os.getenv("LLM_MODEL")
    base_url = os.getenv("LLM_BASE_URL")
    api_key = os.getenv("LLM_API_KEY")
    logger.info("Using model: %s at %s", model, base_url)
    logger.info("Rate limit: %d requests per minute", MAX_REQUESTS_PER_MINUTE)

    client = OpenAI(base_url=base_url, api_key=api_key)
    limiter = RateLimiter(MAX_REQUESTS_PER_MINUTE)

    documents = load_documents()
    if not documents:
        logger.error("No documents found in database. Run ingest_pipeline.py first.")
        return

    logger.info("Loaded %d documents from database", len(documents))

    ground_truth = []

    for doc in tqdm(documents, desc="Generating questions"):
        limiter.wait()
        try:
            records = generate_questions_for_doc(client, doc)
            ground_truth.extend(records)
        except Exception as e:
            logger.error("Failed for doc %s: %s", doc["path"], e)

    if not ground_truth:
        logger.error("No ground truth generated")
        return

    df = pd.DataFrame(ground_truth)
    output_path = "data/ground_truth.csv"
    os.makedirs("data", exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info("Generated %d questions for %d documents", len(ground_truth), len(documents))
    logger.info("Saved to %s", output_path)


if __name__ == "__main__":
    main()


