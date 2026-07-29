from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestReranker:
    @patch("app.retrieval.reranker.ort.InferenceSession")
    @patch("app.retrieval.reranker.Tokenizer")
    def test_init_loads_model(self, mock_tokenizer_cls, mock_session_cls):
        from app.retrieval.reranker import Reranker

        mock_session = MagicMock()
        inp = MagicMock()
        inp.name = "input_ids"
        mock_session.get_inputs.return_value = [inp]
        out = MagicMock()
        out.name = "logits"
        mock_session.get_outputs.return_value = [out]
        mock_session_cls.return_value = mock_session

        mock_tokenizer = MagicMock()
        mock_tokenizer.token_to_id.return_value = 0
        mock_tokenizer_cls.from_file.return_value = mock_tokenizer

        reranker = Reranker(path="models/test-reranker")
        assert reranker.pad_id == 0

    @patch("app.retrieval.reranker.ort.InferenceSession")
    @patch("app.retrieval.reranker.Tokenizer")
    def test_score_returns_float_list(self, mock_tokenizer_cls, mock_session_cls):
        from app.retrieval.reranker import Reranker

        mock_session = MagicMock()
        inp = MagicMock()
        inp.name = "input_ids"
        mock_session.get_inputs.return_value = [inp]
        out = MagicMock()
        out.name = "logits"
        mock_session.get_outputs.return_value = [out]
        mock_session_cls.return_value = mock_session

        mock_tokenizer = MagicMock()
        mock_tokenizer.token_to_id.return_value = 0
        mock_tokenizer_cls.from_file.return_value = mock_tokenizer

        logits = np.array([[2.5], [-1.0], [0.5]])
        mock_session.run.return_value = [logits]

        reranker = Reranker(path="models/test-reranker")
        docs = [
            {"content": "doc1"},
            {"content": "doc2"},
            {"content": "doc3"},
        ]
        scores = reranker.score("query", docs)

        assert len(scores) == 3
        assert all(isinstance(s, float) for s in scores)
        assert scores[0] > scores[2] > scores[1]

    @patch("app.retrieval.reranker.ort.InferenceSession")
    @patch("app.retrieval.reranker.Tokenizer")
    def test_score_empty_documents(self, mock_tokenizer_cls, mock_session_cls):
        from app.retrieval.reranker import Reranker

        mock_session = MagicMock()
        inp = MagicMock()
        inp.name = "input_ids"
        mock_session.get_inputs.return_value = [inp]
        out = MagicMock()
        out.name = "logits"
        mock_session.get_outputs.return_value = [out]
        mock_session_cls.return_value = mock_session

        mock_tokenizer = MagicMock()
        mock_tokenizer.token_to_id.return_value = 0
        mock_tokenizer_cls.from_file.return_value = mock_tokenizer

        reranker = Reranker(path="models/test-reranker")
        assert reranker.score("query", []) == []

    @patch("app.retrieval.reranker.ort.InferenceSession")
    @patch("app.retrieval.reranker.Tokenizer")
    def test_rerank_returns_sorted_by_score(self, mock_tokenizer_cls, mock_session_cls):
        from app.retrieval.reranker import Reranker

        mock_session = MagicMock()
        inp = MagicMock()
        inp.name = "input_ids"
        mock_session.get_inputs.return_value = [inp]
        out = MagicMock()
        out.name = "logits"
        mock_session.get_outputs.return_value = [out]
        mock_session_cls.return_value = mock_session

        mock_tokenizer = MagicMock()
        mock_tokenizer.token_to_id.return_value = 0
        mock_tokenizer_cls.from_file.return_value = mock_tokenizer

        logits = np.array([[1.0], [5.0], [3.0]])
        mock_session.run.return_value = [logits]

        reranker = Reranker(path="models/test-reranker")
        docs = [
            {"id": 1, "content": "low score"},
            {"id": 2, "content": "high score"},
            {"id": 3, "content": "mid score"},
        ]
        result = reranker.rerank("query", docs, top_k=2)

        assert len(result) == 2
        assert result[0]["id"] == 2
        assert result[1]["id"] == 3
        assert result[0]["rerank_score"] > result[1]["rerank_score"]

    @patch("app.retrieval.reranker.ort.InferenceSession")
    @patch("app.retrieval.reranker.Tokenizer")
    def test_rerank_preserves_original_fields(self, mock_tokenizer_cls, mock_session_cls):
        from app.retrieval.reranker import Reranker

        mock_session = MagicMock()
        inp = MagicMock()
        inp.name = "input_ids"
        mock_session.get_inputs.return_value = [inp]
        out = MagicMock()
        out.name = "logits"
        mock_session.get_outputs.return_value = [out]
        mock_session_cls.return_value = mock_session

        mock_tokenizer = MagicMock()
        mock_tokenizer.token_to_id.return_value = 0
        mock_tokenizer_cls.from_file.return_value = mock_tokenizer

        logits = np.array([[2.0]])
        mock_session.run.return_value = [logits]

        reranker = Reranker(path="models/test-reranker")
        docs = [{"id": 1, "path": "test.md", "content": "hello", "distance": 0.5}]
        result = reranker.rerank("query", docs)

        assert result[0]["id"] == 1
        assert result[0]["path"] == "test.md"
        assert result[0]["distance"] == 0.5
        assert "rerank_score" in result[0]

    @patch("app.retrieval.reranker.ort.InferenceSession")
    @patch("app.retrieval.reranker.Tokenizer")
    def test_rerank_truncates_content(self, mock_tokenizer_cls, mock_session_cls):
        from app.retrieval.reranker import Reranker, MAX_CONTENT_CHARS

        mock_session = MagicMock()
        inp = MagicMock()
        inp.name = "input_ids"
        mock_session.get_inputs.return_value = [inp]
        out = MagicMock()
        out.name = "logits"
        mock_session.get_outputs.return_value = [out]
        mock_session_cls.return_value = mock_session

        mock_tokenizer = MagicMock()
        mock_tokenizer.token_to_id.return_value = 0
        mock_tokenizer_cls.from_file.return_value = mock_tokenizer

        logits = np.array([[1.0]])
        mock_session.run.return_value = [logits]

        reranker = Reranker(path="models/test-reranker")
        long_content = "x" * 2000
        docs = [{"content": long_content}]
        reranker.rerank("query", docs)

        args = mock_tokenizer.encode_batch.call_args[0][0]
        assert len(args[0][1]) <= MAX_CONTENT_CHARS

    @patch("app.retrieval.reranker.ort.InferenceSession")
    @patch("app.retrieval.reranker.Tokenizer")
    def test_rerank_empty_documents(self, mock_tokenizer_cls, mock_session_cls):
        from app.retrieval.reranker import Reranker

        mock_session = MagicMock()
        inp = MagicMock()
        inp.name = "input_ids"
        mock_session.get_inputs.return_value = [inp]
        out = MagicMock()
        out.name = "logits"
        mock_session.get_outputs.return_value = [out]
        mock_session_cls.return_value = mock_session

        mock_tokenizer = MagicMock()
        mock_tokenizer.token_to_id.return_value = 0
        mock_tokenizer_cls.from_file.return_value = mock_tokenizer

        reranker = Reranker(path="models/test-reranker")
        assert reranker.rerank("query", []) == []

    @patch("app.retrieval.reranker.ort.InferenceSession")
    @patch("app.retrieval.reranker.Tokenizer")
    def test_sigmoid_when_enabled(self, mock_tokenizer_cls, mock_session_cls):
        from app.retrieval.reranker import Reranker

        mock_session = MagicMock()
        inp = MagicMock()
        inp.name = "input_ids"
        mock_session.get_inputs.return_value = [inp]
        out = MagicMock()
        out.name = "logits"
        mock_session.get_outputs.return_value = [out]
        mock_session_cls.return_value = mock_session

        mock_tokenizer = MagicMock()
        mock_tokenizer.token_to_id.return_value = 0
        mock_tokenizer_cls.from_file.return_value = mock_tokenizer

        logits = np.array([[2.0], [-2.0]])
        mock_session.run.return_value = [logits]

        reranker = Reranker(path="models/test-reranker", use_sigmoid=True)
        scores = reranker.score("query", [{"content": "a"}, {"content": "b"}])

        assert all(0.0 <= s <= 1.0 for s in scores)
        assert scores[0] > scores[1]
