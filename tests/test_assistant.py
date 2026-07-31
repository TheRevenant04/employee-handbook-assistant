import os
from unittest.mock import MagicMock, patch

import pytest


class TestEnvBool:
    def test_env_bool_true_values(self):
        from src.services.rag_service import env_bool

        for val in ["1", "true", "yes", "y", "on", "TRUE", "Yes"]:
            with patch.dict(os.environ, {"TEST_VAR": val}):
                assert env_bool("TEST_VAR") is True

    def test_env_bool_false_values(self):
        from src.services.rag_service import env_bool

        for val in ["0", "false", "no", "n", "off", "", "FALSE", "No"]:
            with patch.dict(os.environ, {"TEST_VAR": val}):
                assert env_bool("TEST_VAR") is False

    def test_env_bool_default(self):
        from src.services.rag_service import env_bool

        with patch.dict(os.environ, {}, clear=True):
            assert env_bool("NONEXISTENT_VAR") is False
            assert env_bool("NONEXISTENT_VAR", default=True) is True

    def test_env_bool_invalid_raises(self):
        from src.services.rag_service import env_bool

        with patch.dict(os.environ, {"TEST_VAR": "maybe"}):
            with pytest.raises(ValueError, match="Invalid boolean env var"):
                env_bool("TEST_VAR")

    def test_env_bool_strips_whitespace(self):
        from src.services.rag_service import env_bool

        with patch.dict(os.environ, {"TEST_VAR": "  true  "}):
            assert env_bool("TEST_VAR") is True


class TestCreateAssistant:
    @patch("src.services.rag_service.Evaluator")
    @patch("src.services.rag_service.MetricsCollector")
    @patch("src.services.rag_service.ChatStore")
    @patch("src.services.rag_service.RAG")
    @patch("src.services.rag_service.get_query_rewriter")
    @patch("src.services.rag_service.get_reranker")
    @patch("src.services.rag_service.get_llm_client")
    def test_create_assistant_wires_all_components(
        self, mock_get_llm, mock_get_reranker, mock_get_rewriter,
        mock_rag_cls, mock_chat_store_cls, mock_metrics_cls, mock_evaluator_cls,
    ):
        from src.services.rag_service import create_assistant

        mock_embedder = MagicMock()
        mock_llm = MagicMock()
        mock_reranker = MagicMock()
        mock_rewriter = MagicMock()

        mock_get_llm.return_value = mock_llm
        mock_get_reranker.return_value = mock_reranker
        mock_get_rewriter.return_value = mock_rewriter

        result = create_assistant(
            embedder=mock_embedder,
            llm_client=mock_llm,
            reranker=mock_reranker,
            query_rewriter=mock_rewriter,
        )

        mock_rag_cls.assert_called_once()
        call_kwargs = mock_rag_cls.call_args[1]
        assert call_kwargs["embedder"] is mock_embedder
        assert call_kwargs["llm_client"] is mock_llm
        assert call_kwargs["reranker"] is mock_reranker
        assert call_kwargs["query_rewriter"] is mock_rewriter

    @patch("src.services.rag_service.Evaluator")
    @patch("src.services.rag_service.MetricsCollector")
    @patch("src.services.rag_service.ChatStore")
    @patch("src.services.rag_service.RAG")
    @patch("src.services.rag_service.get_query_rewriter")
    @patch("src.services.rag_service.get_reranker")
    @patch("src.services.rag_service.get_llm_client")
    @patch("src.services.rag_service.Embedder")
    def test_create_assistant_creates_defaults_when_not_provided(
        self, mock_embedder_cls, mock_get_llm, mock_get_reranker, mock_get_rewriter,
        mock_rag_cls, mock_chat_store_cls, mock_metrics_cls, mock_evaluator_cls,
    ):
        from src.services.rag_service import create_assistant

        mock_get_llm.return_value = MagicMock()
        mock_get_reranker.return_value = MagicMock()
        mock_get_rewriter.return_value = MagicMock()

        result = create_assistant()

        mock_embedder_cls.assert_called_once()
        mock_get_llm.assert_called_once()


class TestGetReranker:
    @patch("src.services.rag_service.Reranker")
    @patch.dict(os.environ, {"RERANKER_ENABLED": "false"})
    def test_returns_none_when_disabled(self, mock_reranker_cls):
        from src.services.rag_service import get_reranker

        result = get_reranker()
        assert result is None
        mock_reranker_cls.assert_not_called()

    @patch("src.services.rag_service.Reranker")
    @patch("src.services.rag_service.os.path.exists", return_value=False)
    @patch.dict(os.environ, {"RERANKER_ENABLED": "true"})
    def test_returns_none_when_model_missing(self, mock_exists, mock_reranker_cls):
        from src.services.rag_service import get_reranker

        result = get_reranker()
        assert result is None

    @patch("src.services.rag_service.Reranker")
    @patch("src.services.rag_service.os.path.exists", return_value=True)
    @patch.dict(os.environ, {"RERANKER_ENABLED": "true"})
    def test_returns_reranker_when_enabled(self, mock_exists, mock_reranker_cls):
        from src.services.rag_service import get_reranker

        mock_reranker_cls.return_value = MagicMock()
        result = get_reranker()
        assert result is not None

    @patch("src.services.rag_service.Reranker", side_effect=Exception("load error"))
    @patch("src.services.rag_service.os.path.exists", return_value=True)
    @patch.dict(os.environ, {"RERANKER_ENABLED": "true"})
    def test_returns_none_on_load_error(self, mock_exists, mock_reranker_cls):
        from src.services.rag_service import get_reranker

        result = get_reranker()
        assert result is None


class TestGetQueryRewriter:
    @patch("src.services.rag_service.QueryRewriter")
    @patch.dict(os.environ, {"QUERY_REWRITER_ENABLED": "false"})
    def test_returns_none_when_disabled(self, mock_rewriter_cls):
        from src.services.rag_service import get_query_rewriter

        result = get_query_rewriter()
        assert result is None

    @patch("src.services.rag_service.QueryRewriter")
    @patch.dict(os.environ, {"QUERY_REWRITER_ENABLED": "true", "LLM_MODEL": "test"})
    def test_returns_rewriter_when_enabled(self, mock_rewriter_cls):
        from src.services.rag_service import get_query_rewriter

        mock_rewriter_cls.return_value = MagicMock()
        result = get_query_rewriter()
        assert result is not None

