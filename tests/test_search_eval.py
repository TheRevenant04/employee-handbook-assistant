import json
from unittest.mock import MagicMock, patch, ANY

import numpy as np
import pandas as pd
import pytest


class TestLoadGroundTruth:
    def test_loads_csv(self):
        from src.evaluation.search_eval import load_ground_truth

        df = pd.DataFrame({
            "question": ["Q1", "Q2"],
            "document": ["doc1.md", "doc2.md"],
        })

        with patch("src.evaluation.search_eval.pd.read_csv", return_value=df):
            result = load_ground_truth("fake.csv")

        assert len(result) == 2
        assert result[0]["question"] == "Q1"
        assert result[1]["document"] == "doc2.md"

    def test_raises_on_missing_columns(self):
        from src.evaluation.search_eval import load_ground_truth

        df = pd.DataFrame({"question": ["Q1"]})

        with patch("src.evaluation.search_eval.pd.read_csv", return_value=df):
            with pytest.raises(ValueError, match="missing required columns"):
                load_ground_truth("fake.csv")

    def test_drops_empty_rows(self):
        from src.evaluation.search_eval import load_ground_truth

        df = pd.DataFrame({
            "question": ["Q1", "", "  ", "Q2"],
            "document": ["doc1.md", "doc2.md", "  ", ""],
        })

        with patch("src.evaluation.search_eval.pd.read_csv", return_value=df):
            result = load_ground_truth("fake.csv")

        assert len(result) == 1
        assert result[0]["question"] == "Q1"


class TestHitRate:
    def test_hit_rate_all_hits(self):
        from src.evaluation.search_eval import hit_rate

        relevance = [[1, 0], [1], [0, 1]]
        assert hit_rate(relevance) == 1.0

    def test_hit_rate_partial(self):
        from src.evaluation.search_eval import hit_rate

        relevance = [[1, 0], [0, 0], [1]]
        assert hit_rate(relevance) == 2 / 3

    def test_hit_rate_empty(self):
        from src.evaluation.search_eval import hit_rate

        assert hit_rate([]) == 0.0

    def test_hit_rate_no_hits(self):
        from src.evaluation.search_eval import hit_rate

        assert hit_rate([[0, 0], [0]]) == 0.0


class TestMRR:
    def test_mrr_all_first_rank(self):
        from src.evaluation.search_eval import mrr

        relevance = [[1], [1, 0], [1, 0, 0]]
        assert mrr(relevance) == 1.0

    def test_mrr_varied_ranks(self):
        from src.evaluation.search_eval import mrr

        relevance = [[1], [0, 1], [0, 0, 1]]
        expected = (1.0 + 1 / 2 + 1 / 3) / 3
        assert mrr(relevance) == pytest.approx(expected)

    def test_mrr_empty(self):
        from src.evaluation.search_eval import mrr

        assert mrr([]) == 0.0

    def test_mrr_no_hits(self):
        from src.evaluation.search_eval import mrr

        relevance = [[0, 0], [0]]
        assert mrr(relevance) == 0.0


class TestComputeRelevanceRow:
    def test_hit_found(self):
        from src.evaluation.search_eval import compute_relevance_row

        def search_fn(q):
            return [{"path": "doc1.md"}, {"path": "doc2.md"}]

        relevance, debug = compute_relevance_row(
            {"question": "q", "document": "doc2.md"}, search_fn
        )

        assert relevance == [0, 1]
        assert debug["hit"] == 1
        assert debug["correct_rank"] == 2

    def test_hit_not_found(self):
        from src.evaluation.search_eval import compute_relevance_row

        def search_fn(q):
            return [{"path": "doc1.md"}, {"path": "doc3.md"}]

        relevance, debug = compute_relevance_row(
            {"question": "q", "document": "doc2.md"}, search_fn
        )

        assert relevance == [0, 0]
        assert debug["hit"] == 0
        assert debug["correct_rank"] is None


class TestComputeRelevanceTotal:
    def test_returns_relevance_and_debug(self):
        from src.evaluation.search_eval import compute_relevance_total

        ground_truth = [
            {"question": "q1", "document": "doc1.md"},
            {"question": "q2", "document": "doc2.md"},
        ]

        def search_fn(q):
            return [{"path": "doc1.md"}, {"path": "doc2.md"}]

        relevance_total, debug_rows = compute_relevance_total(ground_truth, search_fn)

        assert len(relevance_total) == 2
        assert relevance_total[0] == [1, 0]
        assert relevance_total[1] == [0, 1]
        assert debug_rows[0]["hit"] == 1
        assert debug_rows[1]["hit"] == 1


class TestEvaluateMethod:
    def test_returns_metrics(self):
        from src.evaluation.search_eval import evaluate_method

        ground_truth = [
            {"question": "q1", "document": "doc1.md"},
            {"question": "q2", "document": "doc2.md"},
        ]

        def search_fn(q):
            return [{"path": "doc1.md"}]

        metrics, debug = evaluate_method("vector", ground_truth, search_fn)

        assert metrics["method"] == "vector"
        assert metrics["hit_rate"] == 0.5
        assert metrics["mrr"] == 0.5
        assert metrics["questions_evaluated"] == 2
        assert len(debug) == 2


class TestSaveOutputs:
    @patch("src.evaluation.search_eval.pd.DataFrame.to_csv")
    @patch("src.evaluation.search_eval.pd.DataFrame.to_json")
    @patch("src.evaluation.search_eval.os.makedirs")
    def test_saves_outputs(self, mock_makedirs, mock_json, mock_csv):
        from src.evaluation.search_eval import save_outputs

        save_outputs(
            [{"method": "vector", "hit_rate": 0.5, "mrr": 0.5, "questions_evaluated": 1}],
            [{"question": "q1", "hit": 1}],
        )

        assert mock_csv.call_count == 2
        mock_json.assert_called_once()


class TestSearchEvaluator:
    @patch("src.evaluation.search_eval.get_connection")
    def test_vector_search(self, mock_get_conn):
        from src.evaluation.search_eval import SearchEvaluator

        embedder = MagicMock()
        embedder.encode.return_value = np.array([0.1, 0.2, 0.3])
        evaluator = SearchEvaluator(embedder)

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [(1, "doc.md", "content", 0.3)]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_conn.return_value.__enter__.return_value = mock_conn

        results = evaluator.vector_search("test query")
        assert len(results) == 1
        assert results[0]["id"] == 1
        assert results[0]["path"] == "doc.md"
        assert results[0]["method"] == "vector"

    def test_rerank_search_raises_without_reranker(self):
        from src.evaluation.search_eval import SearchEvaluator

        evaluator = SearchEvaluator(MagicMock(), reranker=None)
        with pytest.raises(ValueError, match="Reranker not loaded"):
            evaluator.rerank_search("query")

    @patch("src.evaluation.search_eval.get_connection")
    def test_keyword_search(self, mock_get_conn):
        from src.evaluation.search_eval import SearchEvaluator

        embedder = MagicMock()
        evaluator = SearchEvaluator(embedder)

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [(2, "doc.md", "content", 0.8)]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_conn.return_value.__enter__.return_value = mock_conn

        results = evaluator.keyword_search("test query")
        assert len(results) == 1
        assert results[0]["method"] == "keyword"

    def test_embedding_cache(self):
        from src.evaluation.search_eval import SearchEvaluator

        embedder = MagicMock()
        evaluator = SearchEvaluator(embedder)

        evaluator.get_query_embedding("hello")
        evaluator.get_query_embedding("hello")

        assert embedder.encode.call_count == 1
