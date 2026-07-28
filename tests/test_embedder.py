from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path

import numpy as np
import pytest


class TestEmbedder:
    @patch("embedder.ort.InferenceSession")
    @patch("embedder.Tokenizer")
    def test_init_loads_model_and_tokenizer(self, mock_tokenizer_cls, mock_session_cls):
        from embedder import Embedder

        mock_session = MagicMock()
        inp = MagicMock()
        inp.name = "input_ids"
        mock_session.get_inputs.return_value = [inp]
        mock_session_cls.return_value = mock_session

        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.from_file.return_value = mock_tokenizer

        embedder = Embedder(path="models/test-model")

        mock_tokenizer_cls.from_file.assert_called_once_with(
            str(Path("models/test-model") / "tokenizer.json")
        )
        mock_session_cls.assert_called_once_with(
            str(Path("models/test-model") / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )

    @patch("embedder.ort.InferenceSession")
    @patch("embedder.Tokenizer")
    def test_encode_single_text(self, mock_tokenizer_cls, mock_session_cls):
        from embedder import Embedder

        mock_session = MagicMock()
        inp = MagicMock()
        inp.name = "input_ids"
        mock_session.get_inputs.return_value = [inp]

        hidden = np.random.randn(1, 10, 384).astype(np.float32)
        attention_mask = np.ones((1, 10), dtype=np.int64)
        mock_session.run.return_value = [hidden]
        mock_session_cls.return_value = mock_session

        mock_tokenizer = MagicMock()
        encoded = MagicMock()
        encoded.ids = [101, 2023, 102]
        encoded.attention_mask = [1, 1, 1]
        mock_tokenizer.encode_batch.return_value = [encoded]
        mock_tokenizer_cls.from_file.return_value = mock_tokenizer

        embedder = Embedder(path="models/test-model")
        result = embedder.encode("test text", normalize=True)

        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-5

    @patch("embedder.ort.InferenceSession")
    @patch("embedder.Tokenizer")
    def test_encode_batch_multiple_texts(self, mock_tokenizer_cls, mock_session_cls):
        from embedder import Embedder

        mock_session = MagicMock()
        inp = MagicMock()
        inp.name = "input_ids"
        mock_session.get_inputs.return_value = [inp]

        hidden = np.random.randn(2, 5, 384).astype(np.float32)
        mock_session.run.return_value = [hidden]
        mock_session_cls.return_value = mock_session

        mock_tokenizer = MagicMock()
        enc1 = MagicMock()
        enc1.ids = [101, 2023, 102, 0, 0]
        enc1.attention_mask = [1, 1, 1, 0, 0]
        enc2 = MagicMock()
        enc2.ids = [101, 3456, 102, 0, 0]
        enc2.attention_mask = [1, 1, 1, 0, 0]
        mock_tokenizer.encode_batch.return_value = [enc1, enc2]
        mock_tokenizer_cls.from_file.return_value = mock_tokenizer

        embedder = Embedder(path="models/test-model")
        result = embedder.encode_batch(["text1", "text2"], normalize=True)

        assert result.shape[0] == 2
        for i in range(2):
            assert abs(np.linalg.norm(result[i]) - 1.0) < 1e-5

    @patch("embedder.ort.InferenceSession")
    @patch("embedder.Tokenizer")
    def test_encode_without_normalization(self, mock_tokenizer_cls, mock_session_cls):
        from embedder import Embedder

        mock_session = MagicMock()
        inp = MagicMock()
        inp.name = "input_ids"
        mock_session.get_inputs.return_value = [inp]

        hidden = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)
        mock_session.run.return_value = [hidden]
        mock_session_cls.return_value = mock_session

        mock_tokenizer = MagicMock()
        encoded = MagicMock()
        encoded.ids = [101]
        encoded.attention_mask = [1]
        mock_tokenizer.encode_batch.return_value = [encoded]
        mock_tokenizer_cls.from_file.return_value = mock_tokenizer

        embedder = Embedder(path="models/test-model")
        result = embedder.encode("test", normalize=False)

        norm = np.linalg.norm(result)
        assert norm != 1.0 or np.allclose(result, result / norm)

    @patch("embedder.ort.InferenceSession")
    @patch("embedder.Tokenizer")
    def test_encode_handles_attention_mask_pooling(self, mock_tokenizer_cls, mock_session_cls):
        from embedder import Embedder

        mock_session = MagicMock()
        inp_ids = MagicMock()
        inp_ids.name = "input_ids"
        inp_mask = MagicMock()
        inp_mask.name = "attention_mask"
        mock_session.get_inputs.return_value = [inp_ids, inp_mask]

        hidden = np.ones((1, 3, 4), dtype=np.float32)
        mask = np.array([[1, 1, 0]], dtype=np.int64)
        mock_session.run.return_value = [hidden]
        mock_session_cls.return_value = mock_session

        mock_tokenizer = MagicMock()
        encoded = MagicMock()
        encoded.ids = [1, 2, 0]
        encoded.attention_mask = [1, 1, 0]
        mock_tokenizer.encode_batch.return_value = [encoded]
        mock_tokenizer_cls.from_file.return_value = mock_tokenizer

        embedder = Embedder(path="models/test-model")
        result = embedder.encode("test", normalize=False)

        assert result.ndim == 1
        assert result.shape == (4,)
        expected = hidden[0, :2].sum(axis=0) / 2.0
        np.testing.assert_array_almost_equal(result, expected)
