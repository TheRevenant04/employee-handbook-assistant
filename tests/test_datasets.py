from unittest.mock import MagicMock, patch, ANY

import pytest


class TestGenerateQuestionsForDoc:
    @patch("app.evaluation.datasets.retry_with_backoff")
    def test_generates_questions(self, mock_retry):
        mock_retry.side_effect = lambda fn, **kw: fn()

        from app.evaluation.datasets import generate_questions_for_doc

        client = MagicMock()
        parsed = MagicMock()
        parsed.questions = ["What is leave policy?", "How many days?"]
        client.beta.chat.completions.parse.return_value.choices[0].message.parsed = parsed

        doc = {"path": "leave.md", "content": "Leave policy content"}
        result = generate_questions_for_doc(client, doc)

        assert len(result) == 2
        assert result[0]["question"] == "What is leave policy?"
        assert result[0]["document"] == "leave.md"
        assert result[1]["question"] == "How many days?"

    @patch("app.evaluation.datasets.retry_with_backoff")
    def test_handles_none_response(self, mock_retry):
        mock_retry.side_effect = lambda fn, **kw: fn()

        from app.evaluation.datasets import generate_questions_for_doc

        client = MagicMock()
        client.beta.chat.completions.parse.return_value.choices[0].message.parsed = None

        result = generate_questions_for_doc(client, {"path": "test.md", "content": "test"})
        assert result == []


class TestLoadDocuments:
    @patch("app.evaluation.datasets.get_connection")
    def test_loads_documents(self, mock_get_conn):
        from app.evaluation.datasets import load_documents

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [("path1.md", "content1"), ("path2.md", "content2")]
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        result = load_documents()
        assert len(result) == 2
        assert result[0]["path"] == "path1.md"
        assert result[1]["content"] == "content2"


class TestMain:
    @patch("app.evaluation.datasets.load_documents")
    @patch("app.evaluation.datasets.RateLimiter")
    @patch("app.evaluation.datasets.OpenAI")
    @patch("app.evaluation.datasets.pd.DataFrame.to_csv")
    @patch("app.evaluation.datasets.os.makedirs")
    def test_main_happy_path(self, mock_makedirs, mock_to_csv, mock_openai, mock_limiter, mock_load):
        from app.evaluation.datasets import main

        mock_load.return_value = [{"path": "leave.md", "content": "Leave policy"}]
        mock_openai.return_value = MagicMock()
        mock_limiter.return_value = MagicMock()

        with patch("app.evaluation.datasets.generate_questions_for_doc") as mock_gen:
            mock_gen.return_value = [{"question": "Q?", "document": "leave.md"}]
            main()

        mock_gen.assert_called_once()
        mock_to_csv.assert_called_once()

    @patch("app.evaluation.datasets.load_documents")
    def test_exits_when_no_documents(self, mock_load):
        from app.evaluation.datasets import main

        mock_load.return_value = []
        main()
