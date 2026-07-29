import os
import json
import logging
from pathlib import Path
from typing import Callable, Any

import pandas as pd
from psycopg import sql
from tqdm.auto import tqdm

from app.embeddings.provider import Embedder
from app.retrieval.reranker import Reranker
from app.vectorstore.pgvector_store import get_connection
from app.core.logging import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


TABLE_NAME: str = os.getenv("TABLE_NAME", "employee_handbook")
MODEL_PATH: str = os.getenv("MODEL_PATH", "models/Xenova/all-MiniLM-L6-v2")
RERANKER_MODEL_PATH: str = os.getenv("RERANKER_MODEL_PATH", "models/Xenova/ms-marco-MiniLM-L-6-v2")
NUM_RESULTS: int = int(os.getenv("NUM_RESULTS", "5"))
HYBRID_ALPHAS: list[float] = [
    float(x.strip())
    for x in os.getenv("HYBRID_ALPHAS", "0.2,0.5,0.8").split(",")
    if x.strip()
]
GROUND_TRUTH_PATH: str = os.getenv("GROUND_TRUTH_PATH", "data/ground_truth.csv")
OUTPUT_DIR: Path = Path(os.getenv("EVAL_OUTPUT_DIR", "data/evaluation"))


def load_ground_truth(path: str = GROUND_TRUTH_PATH) -> list[dict[str, Any]]:
    df = pd.read_csv(path)

    required_columns = {"question", "document"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Ground truth file is missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["question", "document"]).copy()
    df["question"] = df["question"].astype(str).str.strip()
    df["document"] = df["document"].astype(str).str.strip()
    df = df[(df["question"] != "") & (df["document"] != "")]

    return df.to_dict(orient="records")


class SearchEvaluator:
    def __init__(self, embedder: Embedder, reranker: Reranker | None = None):
        self.embedder = embedder
        self.reranker = reranker
        self.embedding_cache: dict[str, Any] = {}

    def get_query_embedding(self, query_text: str):
        if query_text not in self.embedding_cache:
            self.embedding_cache[query_text] = self.embedder.encode(query_text, normalize=True)
        return self.embedding_cache[query_text]

    def vector_search(self, query_text: str, num_results: int = NUM_RESULTS) -> list[dict[str, Any]]:
        query_vector = self.get_query_embedding(query_text)

        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        """
                        SELECT id, path, content, embedding <=> %s AS distance
                        FROM {table}
                        ORDER BY embedding <=> %s
                        LIMIT %s
                        """
                    ).format(table=sql.Identifier(TABLE_NAME)),
                    (query_vector, query_vector, num_results),
                )
                rows = cur.fetchall()

            return [
                {
                    "id": row[0],
                    "path": row[1],
                    "content": row[2],
                    "score": float(row[3]),
                    "score_type": "distance",
                    "method": "vector",
                }
                for row in rows
            ]
        finally:
            conn.close()

    def keyword_search(self, query_text: str, num_results: int = NUM_RESULTS) -> list[dict[str, Any]]:
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        """
                        SELECT id, path, content,
                               ts_rank_cd(content_tsv, plainto_tsquery('english', %s)) AS rank
                        FROM {table}
                        WHERE content_tsv @@ plainto_tsquery('english', %s)
                        ORDER BY rank DESC
                        LIMIT %s
                        """
                    ).format(table=sql.Identifier(TABLE_NAME)),
                    (query_text, query_text, num_results),
                )
                rows = cur.fetchall()

            return [
                {
                    "id": row[0],
                    "path": row[1],
                    "content": row[2],
                    "score": float(row[3]),
                    "score_type": "rank",
                    "method": "keyword",
                }
                for row in rows
            ]
        finally:
            conn.close()

    def hybrid_search(
        self,
        query_text: str,
        num_results: int = NUM_RESULTS,
        alpha: float = 0.5,
    ) -> list[dict[str, Any]]:
        query_vector = self.get_query_embedding(query_text)
        fetch_k = max(num_results * 3, 10)

        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        """
                        WITH vector_results AS (
                            SELECT id, path, content,
                                   embedding <=> %s AS v_distance,
                                   ROW_NUMBER() OVER (ORDER BY embedding <=> %s) AS v_rank
                            FROM {table}
                            ORDER BY embedding <=> %s
                            LIMIT %s
                        ),
                        keyword_results AS (
                            SELECT id, path, content,
                                   ts_rank_cd(content_tsv, plainto_tsquery('english', %s)) AS k_score,
                                   ROW_NUMBER() OVER (
                                       ORDER BY ts_rank_cd(content_tsv, plainto_tsquery('english', %s)) DESC
                                   ) AS k_rank
                            FROM {table}
                            WHERE content_tsv @@ plainto_tsquery('english', %s)
                            LIMIT %s
                        ),
                        combined AS (
                            SELECT
                                COALESCE(v.id, k.id) AS id,
                                COALESCE(v.path, k.path) AS path,
                                COALESCE(v.content, k.content) AS content,
                                COALESCE(v.v_rank, %s) AS v_rank,
                                COALESCE(k.k_rank, %s) AS k_rank
                            FROM vector_results v
                            FULL OUTER JOIN keyword_results k ON v.id = k.id
                        )
                        SELECT id, path, content,
                               %s * (1.0 / (1.0 + v_rank)) + (1.0 - %s) * (1.0 / (1.0 + k_rank)) AS score
                        FROM combined
                        ORDER BY score DESC
                        LIMIT %s
                        """
                    ).format(table=sql.Identifier(TABLE_NAME)),
                    (
                        query_vector, query_vector, query_vector, fetch_k,
                        query_text, query_text, query_text, fetch_k,
                        fetch_k, fetch_k,
                        alpha, alpha,
                        num_results,
                    ),
                )
                rows = cur.fetchall()

            return [
                {
                    "id": row[0],
                    "path": row[1],
                    "content": row[2],
                    "score": float(row[3]),
                    "score_type": "hybrid_score",
                    "method": f"hybrid_{alpha}",
                }
                for row in rows
            ]
        finally:
            conn.close()

    def rerank_search(
        self,
        query_text: str,
        num_results: int = NUM_RESULTS,
        alpha: float = 0.5,
    ) -> list[dict[str, Any]]:
        if not self.reranker:
            raise ValueError("Reranker not loaded")

        candidates = self.hybrid_search(query_text, num_results=num_results * 3, alpha=alpha)
        reranked = self.reranker.rerank(query_text, candidates, top_k=num_results)

        for doc in reranked:
            doc["method"] = f"rerank_hybrid_{alpha}"

        return reranked


def compute_relevance_row(
    item: dict[str, Any],
    search_function: Callable[[str], list[dict[str, Any]]],
) -> tuple[list[int], dict[str, Any]]:
    expected_path = item["document"]
    question = item["question"]
    results = search_function(question)

    relevance = [int(result["path"] == expected_path) for result in results]

    correct_rank = None
    for idx, value in enumerate(relevance, start=1):
        if value == 1:
            correct_rank = idx
            break

    debug_row = {
        "question": question,
        "expected_document": expected_path,
        "hit": int(correct_rank is not None),
        "correct_rank": correct_rank,
        "top_result": results[0]["path"] if results else None,
        "returned_paths": json.dumps([r["path"] for r in results]),
    }
    return relevance, debug_row


def compute_relevance_total(
    ground_truth: list[dict[str, Any]],
    search_function: Callable[[str], list[dict[str, Any]]],
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    relevance_total = []
    debug_rows = []

    for item in tqdm(ground_truth, desc="Computing relevance"):
        relevance, debug_row = compute_relevance_row(item, search_function)
        relevance_total.append(relevance)
        debug_rows.append(debug_row)

    return relevance_total, debug_rows


def hit_rate(relevance: list[list[int]]) -> float:
    if not relevance:
        return 0.0
    hits = sum(1 for row in relevance if 1 in row)
    return hits / len(relevance)


def mrr(relevance: list[list[int]]) -> float:
    if not relevance:
        return 0.0

    total_score = 0.0
    for row in relevance:
        for rank, value in enumerate(row, start=1):
            if value == 1:
                total_score += 1.0 / rank
                break
    return total_score / len(relevance)


def evaluate_method(
    method_name: str,
    ground_truth: list[dict[str, Any]],
    search_function: Callable[[str], list[dict[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    relevance_total, debug_rows = compute_relevance_total(ground_truth, search_function)

    metrics = {
        "method": method_name,
        "hit_rate": hit_rate(relevance_total),
        "mrr": mrr(relevance_total),
        "questions_evaluated": len(ground_truth),
    }

    for row in debug_rows:
        row["method"] = method_name

    return metrics, debug_rows


def save_outputs(summary_rows: list[dict[str, Any]], debug_rows: list[dict[str, Any]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summary_rows).sort_values("mrr", ascending=False)
    debug_df = pd.DataFrame(debug_rows)

    summary_path = OUTPUT_DIR / "evaluation_summary.csv"
    debug_path = OUTPUT_DIR / "evaluation_debug.csv"
    json_path = OUTPUT_DIR / "evaluation_summary.json"

    summary_df.to_csv(summary_path, index=False)
    debug_df.to_csv(debug_path, index=False)
    summary_df.to_json(json_path, orient="records", indent=2)

    logger.info("Saved summary to %s", summary_path)
    logger.info("Saved debug rows to %s", debug_path)
    logger.info("Saved JSON summary to %s", json_path)


def main():
    logger.info("=== Search Evaluation ===")
    logger.info("Ground truth: %s", GROUND_TRUTH_PATH)
    logger.info("Model path: %s", MODEL_PATH)
    logger.info("Reranker model path: %s", RERANKER_MODEL_PATH)
    logger.info("Table name: %s", TABLE_NAME)
    logger.info("Top-k: %d", NUM_RESULTS)
    logger.info("Hybrid alphas: %s", HYBRID_ALPHAS)

    ground_truth = load_ground_truth()
    if not ground_truth:
        logger.error("No ground truth rows found.")
        return

    logger.info("Loaded %d ground truth questions", len(ground_truth))

    embedder = Embedder(MODEL_PATH)

    reranker = None
    if os.path.exists(RERANKER_MODEL_PATH):
        try:
            reranker = Reranker(RERANKER_MODEL_PATH)
            logger.info("Loaded reranker from %s", RERANKER_MODEL_PATH)
        except Exception:
            logger.warning("Failed to load reranker from %s", RERANKER_MODEL_PATH)
    else:
        logger.info("Reranker model not found at %s, skipping reranker evaluation", RERANKER_MODEL_PATH)

    evaluator = SearchEvaluator(embedder, reranker=reranker)

    summary_rows = []
    debug_rows = []

    logger.info("Evaluating vector search...")
    vector_metrics, vector_debug = evaluate_method(
        "vector",
        ground_truth,
        lambda query: evaluator.vector_search(query, NUM_RESULTS),
    )
    summary_rows.append(vector_metrics)
    debug_rows.extend(vector_debug)

    logger.info("Evaluating keyword search...")
    keyword_metrics, keyword_debug = evaluate_method(
        "keyword",
        ground_truth,
        lambda query: evaluator.keyword_search(query, NUM_RESULTS),
    )
    summary_rows.append(keyword_metrics)
    debug_rows.extend(keyword_debug)

    for alpha in HYBRID_ALPHAS:
        method_name = f"hybrid_{alpha}"
        logger.info("Evaluating %s...", method_name)

        hybrid_metrics, hybrid_debug = evaluate_method(
            method_name,
            ground_truth,
            lambda query, a=alpha: evaluator.hybrid_search(query, NUM_RESULTS, alpha=a),
        )
        summary_rows.append(hybrid_metrics)
        debug_rows.extend(hybrid_debug)

    if reranker:
        for alpha in HYBRID_ALPHAS:
            method_name = f"rerank_hybrid_{alpha}"
            logger.info("Evaluating %s...", method_name)

            rerank_metrics, rerank_debug = evaluate_method(
                method_name,
                ground_truth,
                lambda query, a=alpha: evaluator.rerank_search(query, NUM_RESULTS, alpha=a),
            )
            summary_rows.append(rerank_metrics)
            debug_rows.extend(rerank_debug)

    save_outputs(summary_rows, debug_rows)

    summary_df = pd.DataFrame(summary_rows).sort_values("mrr", ascending=False)

    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print()

    best = summary_df.iloc[0]
    print(
        f"Best method: {best['method']} "
        f"(MRR={best['mrr']:.3f}, Hit Rate={best['hit_rate']:.3f})"
    )


if __name__ == "__main__":
    main()