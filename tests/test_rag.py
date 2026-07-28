from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from rag import RAG


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


class TestRAGInit:
    def test_init_with_defaults(self, mock_embedder, mock_llm_client, mock_chat_store, mock_metrics):
        rag = RAG(
            embedder=mock_embedder,
            llm_client=mock_llm_client,
            chat_store=mock_chat_store,
            metrics=mock_metrics,
        )
        assert rag.embedder is mock_embedder
        assert rag.llm_client is mock_llm_client
        assert rag.chat_store is mock_chat_store
        assert rag.metrics is mock_metrics

    def test_init_with_custom_instructions(self, mock_embedder, mock_llm_client, mock_chat_store, mock_metrics):
        custom = "Custom instructions"
        rag = RAG(
            embedder=mock_embedder,
            llm_client=mock_llm_client,
            chat_store=mock_chat_store,
            instructions=custom,
            metrics=mock_metrics,
        )
        assert rag.instructions == custom

    def test_init_with_model_override(self, mock_embedder, mock_llm_client, mock_chat_store, mock_metrics):
        rag = RAG(
            embedder=mock_embedder,
            llm_client=mock_llm_client,
            chat_store=mock_chat_store,
            model="custom-model",
            metrics=mock_metrics,
        )
        assert rag.model == "custom-model"


class TestRAGBuildContext:
    def test_build_context_joins_content(self, mock_embedder, mock_llm_client, mock_chat_store, mock_metrics):
        rag = RAG(
            embedder=mock_embedder,
            llm_client=mock_llm_client,
            chat_store=mock_chat_store,
            metrics=mock_metrics,
        )
        results = [
            {"content": "First section"},
            {"content": "Second section"},
        ]
        context = rag.build_context(results)
        assert "First section" in context
        assert "Second section" in context

    def test_build_context_empty(self, mock_embedder, mock_llm_client, mock_chat_store, mock_metrics):
        rag = RAG(
            embedder=mock_embedder,
            llm_client=mock_llm_client,
            chat_store=mock_chat_store,
            metrics=mock_metrics,
        )
        assert rag.build_context([]) == ""

    def test_build_context_missing_content(self, mock_embedder, mock_llm_client, mock_chat_store, mock_metrics):
        rag = RAG(
            embedder=mock_embedder,
            llm_client=mock_llm_client,
            chat_store=mock_chat_store,
            metrics=mock_metrics,
        )
        results = [{"id": 1}, {"content": "has content"}]
        context = rag.build_context(results)
        assert "has content" in context


class TestRAGBuildPrompt:
    def test_build_prompt_includes_question_and_context(
        self, mock_embedder, mock_llm_client, mock_chat_store, mock_metrics
    ):
        rag = RAG(
            embedder=mock_embedder,
            llm_client=mock_llm_client,
            chat_store=mock_chat_store,
            metrics=mock_metrics,
        )
        results = [{"content": "policy content here"}]
        prompt = rag.build_prompt("What is the leave policy?", results)

        assert "What is the leave policy?" in prompt
        assert "policy content here" in prompt

    def test_build_prompt_empty_context(self, mock_embedder, mock_llm_client, mock_chat_store, mock_metrics):
        rag = RAG(
            embedder=mock_embedder,
            llm_client=mock_llm_client,
            chat_store=mock_chat_store,
            metrics=mock_metrics,
        )
        prompt = rag.build_prompt("question", [])
        assert "question" in prompt


class TestRAGLLM:
    def test_llm_returns_text_and_usage(self, mock_embedder, mock_llm_client, mock_chat_store, mock_metrics):
        rag = RAG(
            embedder=mock_embedder,
            llm_client=mock_llm_client,
            chat_store=mock_chat_store,
            cost_per_input_token=0.001,
            cost_per_output_token=0.002,
            metrics=mock_metrics,
        )
        result = rag.llm("test prompt")

        assert result["text"] == "This is a test answer."
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50
        assert result["cost"] == pytest.approx(0.1 + 0.1)

    def test_llm_with_zero_cost(self, mock_embedder, mock_chat_store, mock_metrics):
        mock_client = MagicMock()
        choice = MagicMock()
        choice.message.content = "answer"
        usage = MagicMock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 5
        response = MagicMock()
        response.choices = [choice]
        response.usage = usage
        mock_client.chat.completions.create.return_value = response

        rag = RAG(
            embedder=mock_embedder,
            llm_client=mock_client,
            chat_store=mock_chat_store,
            cost_per_input_token=0,
            cost_per_output_token=0,
            metrics=mock_metrics,
        )
        result = rag.llm("prompt")
        assert result["cost"] == 0

    def test_llm_handles_no_usage(self, mock_embedder, mock_chat_store, mock_metrics):
        mock_client = MagicMock()
        choice = MagicMock()
        choice.message.content = "answer"
        response = MagicMock()
        response.choices = [choice]
        response.usage = None
        mock_client.chat.completions.create.return_value = response

        rag = RAG(
            embedder=mock_embedder,
            llm_client=mock_client,
            chat_store=mock_chat_store,
            metrics=mock_metrics,
        )
        result = rag.llm("prompt")
        assert result["input_tokens"] == 0
        assert result["output_tokens"] == 0

    def test_llm_handles_no_choices(self, mock_embedder, mock_chat_store, mock_metrics):
        mock_client = MagicMock()
        response = MagicMock()
        response.choices = []
        response.usage = MagicMock()
        response.usage.prompt_tokens = 0
        response.usage.completion_tokens = 0
        mock_client.chat.completions.create.return_value = response

        rag = RAG(
            embedder=mock_embedder,
            llm_client=mock_client,
            chat_store=mock_chat_store,
            metrics=mock_metrics,
        )
        result = rag.llm("prompt")
        assert result["text"] == ""

    def test_llm_passes_model_and_messages(self, mock_embedder, mock_chat_store, mock_metrics):
        mock_client = MagicMock()
        choice = MagicMock()
        choice.message.content = "answer"
        usage = MagicMock()
        usage.prompt_tokens = 0
        usage.completion_tokens = 0
        response = MagicMock()
        response.choices = [choice]
        response.usage = usage
        mock_client.chat.completions.create.return_value = response

        rag = RAG(
            embedder=mock_embedder,
            llm_client=mock_client,
            chat_store=mock_chat_store,
            model="my-model",
            instructions="system prompt",
            metrics=mock_metrics,
        )
        rag.llm("user prompt")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "my-model"
        assert call_kwargs["messages"][0]["content"] == "system prompt"
        assert call_kwargs["messages"][1]["content"] == "user prompt"


class TestRAGVectorSearch:
    @patch("rag.get_connection")
    def test_vector_search_returns_results(self, mock_get_conn, mock_embedder, mock_chat_store, mock_metrics):
        conn_cm, mock_cursor = _make_db_mock([
            (1, "test.md", "content", 0.3),
            (2, "test2.md", "content2", 0.5),
        ])
        mock_get_conn.return_value = conn_cm

        rag = RAG(
            embedder=mock_embedder,
            llm_client=MagicMock(),
            chat_store=mock_chat_store,
            metrics=mock_metrics,
        )
        results = rag.vector_search("query", num_results=2)

        assert len(results) == 2
        assert results[0]["id"] == 1
        assert results[0]["distance"] == 0.3
        mock_embedder.encode.assert_called_once_with("query", normalize=True)


class TestRAGHybridSearch:
    @patch("rag.get_connection")
    def test_hybrid_search_returns_results(self, mock_get_conn, mock_embedder, mock_chat_store, mock_metrics):
        conn_cm, mock_cursor = _make_db_mock([
            (1, "test.md", "content", 0.8),
        ])
        mock_get_conn.return_value = conn_cm

        rag = RAG(
            embedder=mock_embedder,
            llm_client=MagicMock(),
            chat_store=mock_chat_store,
            metrics=mock_metrics,
        )
        results = rag.hybrid_search("query", num_results=5, alpha=0.5)

        assert len(results) == 1
        assert results[0]["id"] == 1


class TestRAGSearch:
    def test_search_calls_hybrid_search(self, mock_embedder, mock_chat_store, mock_metrics):
        rag = RAG(
            embedder=mock_embedder,
            llm_client=MagicMock(),
            chat_store=mock_chat_store,
            metrics=mock_metrics,
        )
        with patch.object(rag, "hybrid_search", return_value=[{"id": 1}]) as mock_hybrid:
            result = rag.search("query")
            mock_hybrid.assert_called_once_with("query", 5)

    def test_search_with_reranker(self, mock_embedder, mock_chat_store, mock_metrics, mock_reranker):
        rag = RAG(
            embedder=mock_embedder,
            llm_client=MagicMock(),
            chat_store=mock_chat_store,
            metrics=mock_metrics,
            reranker=mock_reranker,
        )
        with patch.object(rag, "hybrid_search", return_value=[{"id": 1}]):
            result = rag.search("query", num_results=3)
            mock_reranker.rerank.assert_called_once_with("query", [{"id": 1}], top_k=3)

    def test_search_without_results_skips_reranker(
        self, mock_embedder, mock_chat_store, mock_metrics, mock_reranker
    ):
        rag = RAG(
            embedder=mock_embedder,
            llm_client=MagicMock(),
            chat_store=mock_chat_store,
            metrics=mock_metrics,
            reranker=mock_reranker,
        )
        with patch.object(rag, "hybrid_search", return_value=[]):
            result = rag.search("query")
            mock_reranker.rerank.assert_not_called()


class TestRAGRag:
    @patch("rag.get_connection")
    def test_rag_full_pipeline(self, mock_get_conn, mock_embedder, mock_chat_store, mock_metrics):
        conn_cm, mock_cursor = _make_db_mock([(1, "path", "content", 0.3)])
        mock_get_conn.return_value = conn_cm

        mock_llm = MagicMock()
        choice = MagicMock()
        choice.message.content = "The answer is yes."
        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 20
        response = MagicMock()
        response.choices = [choice]
        response.usage = usage
        mock_llm.chat.completions.create.return_value = response

        rag = RAG(
            embedder=mock_embedder,
            llm_client=mock_llm,
            chat_store=mock_chat_store,
            metrics=mock_metrics,
        )
        result = rag.rag("What is the policy?", conversation_id=1)

        assert result["answer"] == "The answer is yes."
        assert result["query"] == "What is the policy?"
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 20
        mock_chat_store.add_message.assert_called_once()

    @patch("rag.get_connection")
    def test_rag_with_query_rewriter(
        self, mock_get_conn, mock_embedder, mock_chat_store, mock_metrics, mock_query_rewriter
    ):
        conn_cm, mock_cursor = _make_db_mock([])
        mock_get_conn.return_value = conn_cm

        mock_llm = MagicMock()
        choice = MagicMock()
        choice.message.content = "I don't know."
        usage = MagicMock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 5
        response = MagicMock()
        response.choices = [choice]
        response.usage = usage
        mock_llm.chat.completions.create.return_value = response

        rag = RAG(
            embedder=mock_embedder,
            llm_client=mock_llm,
            chat_store=mock_chat_store,
            metrics=mock_metrics,
            query_rewriter=mock_query_rewriter,
        )
        rag.rag("tell me about leave", conversation_id=1)

        mock_query_rewriter.rewrite.assert_called_once_with("tell me about leave")

    @patch("rag.get_connection")
    def test_rag_records_metrics_on_success(self, mock_get_conn, mock_embedder, mock_chat_store, mock_metrics):
        conn_cm, mock_cursor = _make_db_mock([(1, "path", "content", 0.3)])
        mock_get_conn.return_value = conn_cm

        mock_llm = MagicMock()
        choice = MagicMock()
        choice.message.content = "answer"
        usage = MagicMock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 5
        response = MagicMock()
        response.choices = [choice]
        response.usage = usage
        mock_llm.chat.completions.create.return_value = response

        rag = RAG(
            embedder=mock_embedder,
            llm_client=mock_llm,
            chat_store=mock_chat_store,
            metrics=mock_metrics,
        )
        rag.rag("query", conversation_id=1)

        mock_chat_store.record_metrics.assert_called_once()
        call_kwargs = mock_chat_store.record_metrics.call_args[1]
        assert call_kwargs["success"] is True

    @patch("rag.get_connection")
    def test_rag_records_error_on_failure(self, mock_get_conn, mock_embedder, mock_chat_store, mock_metrics):
        conn_cm, mock_cursor = _make_db_mock([])
        mock_cursor.fetchall.side_effect = Exception("DB error")
        mock_get_conn.return_value = conn_cm

        rag = RAG(
            embedder=mock_embedder,
            llm_client=MagicMock(),
            chat_store=mock_chat_store,
            metrics=mock_metrics,
        )

        with pytest.raises(Exception, match="DB error"):
            rag.rag("query", conversation_id=1)

        mock_metrics.record_error.assert_called_once()
        mock_chat_store.record_metrics.assert_called_once()
        call_kwargs = mock_chat_store.record_metrics.call_args[1]
        assert call_kwargs["success"] is False

    @patch("rag.get_connection")
    def test_rag_calls_evaluator(self, mock_get_conn, mock_embedder, mock_chat_store, mock_metrics, mock_evaluator):
        conn_cm, mock_cursor = _make_db_mock([(1, "path", "content", 0.3)])
        mock_get_conn.return_value = conn_cm

        mock_llm = MagicMock()
        choice = MagicMock()
        choice.message.content = "answer"
        usage = MagicMock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 5
        response = MagicMock()
        response.choices = [choice]
        response.usage = usage
        mock_llm.chat.completions.create.return_value = response

        rag = RAG(
            embedder=mock_embedder,
            llm_client=mock_llm,
            chat_store=mock_chat_store,
            metrics=mock_metrics,
            evaluator=mock_evaluator,
        )
        rag.rag("query", conversation_id=1)

        mock_evaluator.evaluate.assert_called_once()

    @patch("rag.get_connection")
    def test_rag_skips_evaluator_on_failure(
        self, mock_get_conn, mock_embedder, mock_chat_store, mock_metrics, mock_evaluator
    ):
        conn_cm, mock_cursor = _make_db_mock([])
        mock_cursor.fetchall.side_effect = Exception("fail")
        mock_get_conn.return_value = conn_cm

        rag = RAG(
            embedder=mock_embedder,
            llm_client=MagicMock(),
            chat_store=mock_chat_store,
            metrics=mock_metrics,
            evaluator=mock_evaluator,
        )

        with pytest.raises(Exception):
            rag.rag("query", conversation_id=1)

        mock_evaluator.evaluate.assert_not_called()
