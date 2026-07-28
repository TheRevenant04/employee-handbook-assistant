from unittest.mock import MagicMock, patch, mock_open

import pytest

import ingest_pipeline


class TestFetchMarkdownFiles:
    @patch("ingest_pipeline.requests.Session")
    def test_fetches_markdown_files(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        tree_response = MagicMock()
        tree_response.json.return_value = {
            "tree": [
                {"type": "blob", "path": "policies/leave.md"},
                {"type": "blob", "path": "README.md"},
                {"type": "blob", "path": "image.png"},
            ]
        }
        tree_response.raise_for_status = MagicMock()

        file_response = MagicMock()
        file_response.text = "# Leave Policy"
        file_response.raise_for_status = MagicMock()

        mock_session.get.side_effect = [tree_response, file_response, file_response]

        files = ingest_pipeline.fetch_markdown_files()

        assert len(files) == 2
        assert files[0]["path"] == "policies/leave.md"
        assert files[1]["path"] == "README.md"

    @patch("ingest_pipeline.requests.Session")
    def test_returns_empty_when_no_markdown(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        tree_response = MagicMock()
        tree_response.json.return_value = {"tree": [{"type": "blob", "path": "image.png"}]}
        tree_response.raise_for_status = MagicMock()
        mock_session.get.return_value = tree_response

        files = ingest_pipeline.fetch_markdown_files()
        assert files == []

    @patch("ingest_pipeline.requests.Session")
    def test_raises_on_api_error(self, mock_session_cls):
        import requests

        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_session.get.side_effect = requests.exceptions.ConnectionError("timeout")

        with pytest.raises(requests.exceptions.ConnectionError):
            ingest_pipeline.fetch_markdown_files()


class TestEmbedAndStore:
    @patch("embedder.Embedder")
    def test_embeds_and_stores_documents(self, mock_embedder_cls):
        import numpy as np

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        mock_embedder = MagicMock()
        mock_embedder.encode_batch.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_embedder_cls.return_value = mock_embedder

        documents = [
            {"path": "a.md", "content": "content a"},
            {"path": "b.md", "content": "content b"},
        ]

        ingest_pipeline.embed_and_store(mock_conn, documents)

        mock_embedder.encode_batch.assert_called_once()
        mock_cursor.executemany.assert_called_once()
        mock_conn.commit.assert_called_once()

    @patch("embedder.Embedder")
    def test_handles_empty_documents(self, mock_embedder_cls):
        import numpy as np

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        mock_embedder = MagicMock()
        mock_embedder.encode_batch.return_value = np.array([]).reshape(0, 384)
        mock_embedder_cls.return_value = mock_embedder

        ingest_pipeline.embed_and_store(mock_conn, [])

        mock_conn.commit.assert_called_once()


class TestInitDb:
    def test_creates_table_and_indexes(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        ingest_pipeline.init_db(mock_conn)

        assert mock_cursor.execute.call_count >= 4
        mock_conn.commit.assert_called_once()
