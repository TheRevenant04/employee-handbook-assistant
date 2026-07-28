from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag import RAG
from assistant import create_assistant


def _make_db_mock(fetchall_return):
    """Create a properly chained DB mock for with get_connection() as conn: with conn.cursor() as cur: patterns."""
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = fetchall_return

    cursor_cm = MagicMock()
    cursor_cm.__enter__ = MagicMock(return_value=mock_cursor)
    cursor_cm.__exit__ = MagicMock(return_value=False)

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = cursor_cm

    conn_cm = MagicMock()
    conn_cm.__enter__ = MagicMock(return_value=mock_conn)
    conn_cm.__exit__ = MagicMock(return_value=False)

    return conn_cm, mock_cursor


class TestRAGIntegration:
    """Integration tests that wire multiple components together with mocks."""

    def _make_rag(self, **overrides):
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_embedder.encode_batch.return_value = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

        mock_llm = MagicMock()
        choice = MagicMock()
        choice.message.content = "Based on the handbook, you get 25 days of annual leave."
        usage = MagicMock()
        usage.prompt_tokens = 150
        usage.completion_tokens = 30
        response = MagicMock()
        response.choices = [choice]
        response.usage = usage
        mock_llm.chat.completions.create.return_value = response

        mock_chat_store = MagicMock()
        mock_chat_store.add_message.return_value = 1

        mock_metrics = MagicMock()

        class FakeTimer:
            def __init__(self):
                self.result = {"elapsed_ms": 250.0}

            def __enter__(self):
                return self.result

            def __exit__(self, *args):
                pass

        mock_metrics.timer = FakeTimer

        defaults = dict(
            embedder=mock_embedder,
            llm_client=mock_llm,
            chat_store=mock_chat_store,
            metrics=mock_metrics,
        )
        defaults.update(overrides)
        return RAG(**defaults), mock_llm, mock_chat_store, mock_metrics

    @patch("rag.get_connection")
    def test_end_to_end_search_and_llm(self, mock_get_conn):
        conn_cm, _ = _make_db_mock([
            (1, "policies/leave.md", "Annual leave is 25 days.", 0.2),
            (2, "policies/benefits.md", "Benefits include health insurance.", 0.4),
        ])
        mock_get_conn.return_value = conn_cm

        rag, mock_llm, mock_chat_store, _ = self._make_rag()

        result = rag.rag("How many leave days do I get?", conversation_id=1)

        assert result["answer"] == "Based on the handbook, you get 25 days of annual leave."
        assert result["input_tokens"] == 150
        assert result["output_tokens"] == 30
        assert result["num_results"] == 2

        mock_chat_store.add_message.assert_called_once()
        mock_chat_store.record_metrics.assert_called_once()

    @patch("rag.get_connection")
    def test_search_results_used_in_prompt(self, mock_get_conn):
        conn_cm, _ = _make_db_mock([
            (1, "policies/leave.md", "Annual leave is 25 days.", 0.2),
        ])
        mock_get_conn.return_value = conn_cm

        rag, mock_llm, _, _ = self._make_rag()

        rag.rag("leave days", conversation_id=1)

        prompt_used = mock_llm.chat.completions.create.call_args[1]["messages"][1]["content"]
        assert "Annual leave is 25 days." in prompt_used
        assert "leave days" in prompt_used

    @patch("rag.get_connection")
    def test_reranker_reorders_results(self, mock_get_conn):
        conn_cm, _ = _make_db_mock([
            (1, "a.md", "low relevance", 0.8),
            (2, "b.md", "high relevance", 0.2),
        ])
        mock_get_conn.return_value = conn_cm

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [
            {"id": 2, "path": "b.md", "content": "high relevance", "distance": 0.2, "rerank_score": 0.95},
            {"id": 1, "path": "a.md", "content": "low relevance", "distance": 0.8, "rerank_score": 0.3},
        ]

        rag, mock_llm, _, _ = self._make_rag(reranker=mock_reranker)
        rag.rag("query", conversation_id=1)

        prompt_used = mock_llm.chat.completions.create.call_args[1]["messages"][1]["content"]
        assert "high relevance" in prompt_used

    @patch("rag.get_connection")
    def test_query_rewriter_modifies_query(self, mock_get_conn):
        conn_cm, _ = _make_db_mock([
            (1, "a.md", "content", 0.3),
        ])
        mock_get_conn.return_value = conn_cm

        mock_rewriter = MagicMock()
        mock_rewriter.rewrite.return_value = "formal query about annual leave policy"

        rag, _, _, _ = self._make_rag(query_rewriter=mock_rewriter)
        rag.rag("how many days off lol", conversation_id=1)

        mock_rewriter.rewrite.assert_called_once_with("how many days off lol")

    @patch("rag.get_connection")
    def test_metrics_recorded_with_correct_values(self, mock_get_conn):
        conn_cm, _ = _make_db_mock([
            (1, "a.md", "content", 0.3),
        ])
        mock_get_conn.return_value = conn_cm

        rag, _, mock_chat_store, _ = self._make_rag()
        rag.rag("query", conversation_id=42)

        metrics_call = mock_chat_store.record_metrics.call_args[1]
        assert metrics_call["message_id"] == 1
        assert metrics_call["total_latency_ms"] == 250.0
        assert metrics_call["success"] is True
        assert metrics_call["model"] is not None

    @patch("rag.get_connection")
    def test_empty_search_results(self, mock_get_conn):
        conn_cm, _ = _make_db_mock([])
        mock_get_conn.return_value = conn_cm

        rag, mock_llm, mock_chat_store, _ = self._make_rag()
        result = rag.rag("completely unrelated query", conversation_id=1)

        assert result["num_results"] == 0
        prompt_used = mock_llm.chat.completions.create.call_args[1]["messages"][1]["content"]
        assert "completely unrelated query" in prompt_used


class TestAssistantIntegration:
    """Test the create_assistant factory wires everything correctly."""

    @patch("assistant.Evaluator")
    @patch("assistant.MetricsCollector")
    @patch("assistant.ChatStore")
    @patch("assistant.RAG")
    @patch("assistant.get_query_rewriter", return_value=None)
    @patch("assistant.get_reranker", return_value=None)
    @patch("assistant.get_llm_client")
    def test_factory_returns_wired_rag(
        self, mock_get_llm, mock_get_reranker, mock_get_rewriter,
        mock_rag_cls, mock_chat_store_cls, mock_metrics_cls, mock_evaluator_cls,
    ):
        from embedder import Embedder

        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        with patch.object(Embedder, "__init__", return_value=None):
            embedder = Embedder.__new__(Embedder)

        result = create_assistant(
            embedder=embedder,
            llm_client=mock_llm,
        )

        assert result is mock_rag_cls.return_value
        mock_rag_cls.assert_called_once()
        call_kwargs = mock_rag_cls.call_args[1]
        assert call_kwargs["embedder"] is embedder
        assert call_kwargs["llm_client"] is mock_llm
        assert call_kwargs["reranker"] is None
        assert call_kwargs["query_rewriter"] is None
