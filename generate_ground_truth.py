import os
import json
import time
import logging
import collections
import pandas as pd
from pydantic import BaseModel
from openai import OpenAI
from psycopg import sql
from tqdm.auto import tqdm

from db import get_connection
from logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

TABLE_NAME = os.getenv("TABLE_NAME", "employee_handbook")
NUM_QUESTIONS_PER_DOC = int(os.getenv("NUM_QUESTIONS_PER_DOC", "5"))
MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "10"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
RETRY_BASE_DELAY = float(os.getenv("RETRY_BASE_DELAY", "10"))


class Questions(BaseModel):
    questions: list[str]


DATA_GEN_INSTRUCTIONS = """
You emulate an employee who's reading the company handbook.
Formulate 5 questions this employee might ask based on a handbook document.
The document should contain the answer to the questions, and the questions should be complete and not too short.
If possible, use as fewer words as possible from the document.
The output should resemble how people ask questions on the internet.
Not too formal, not too short, not too long.
""".strip()


class RateLimiter:
    def __init__(self, max_requests_per_minute):
        self.max_requests = max_requests_per_minute
        self.window = 60.0
        self.timestamps = collections.deque()

    def wait(self):
        now = time.monotonic()
        while self.timestamps and self.timestamps[0] <= now - self.window:
            self.timestamps.popleft()

        if len(self.timestamps) >= self.max_requests:
            sleep_until = self.timestamps[0] + self.window
            sleep_time = sleep_until - now
            if sleep_time > 0:
                logger.info("Rate limit: sleeping %.1fs", sleep_time)
                time.sleep(sleep_time)

        self.timestamps.append(time.monotonic())


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

    for attempt in range(MAX_RETRIES):
        try:
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

        except Exception as e:
            is_rate_limit = "429" in str(e) or "rate" in str(e).lower()
            if is_rate_limit and attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "Rate limited on doc %s (attempt %d/%d), retrying in %.0fs...",
                    doc["path"], attempt + 1, MAX_RETRIES, delay,
                )
                time.sleep(delay)
            else:
                raise


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
