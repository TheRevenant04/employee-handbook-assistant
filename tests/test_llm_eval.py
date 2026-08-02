from unittest.mock import MagicMock, patch, ANY

import pytest


class TestJudgeAnswer:
    @patch("src.evaluation.llm_eval.retry_with_backoff")
    def test_judge_answer_success(self, mock_retry):
        mock_retry.side_effect = lambda fn, **kw: fn()

        from src.evaluation.llm_eval import judge_answer
        import src.evaluation.llm_eval as llm_eval

        from src.domain.evaluation import EvaluationScores
        mock_scores = EvaluationScores(
            faithfulness_score=4, faithfulness_reasoning="good",
            context_relevance_score=3, context_relevance_reasoning="ok",
            completeness_score=5, completeness_reasoning="great",
        )

        client = MagicMock()
        response = client.beta.chat.completions.parse.return_value
        response.choices[0].message.parsed = mock_scores
        response.usage.prompt_tokens = 100
        response.usage.completion_tokens = 50
        limiter = MagicMock()

        result = judge_answer(client, "question", "answer", "context", "expected doc", limiter)

        assert result is not None
        assert result.scores.faithfulness_score == 4
        assert result.scores.completeness_score == 5
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.cost == (
            100 * llm_eval.JUDGE_COST_PER_INPUT_TOKEN
            + 50 * llm_eval.JUDGE_COST_PER_OUTPUT_TOKEN
        )
        limiter.wait.assert_called_once()

    @patch("src.evaluation.llm_eval.retry_with_backoff")
    def test_judge_answer_returns_none_on_failure(self, mock_retry):
        mock_retry.side_effect = Exception("failed")

        from src.evaluation.llm_eval import judge_answer

        client = MagicMock()
        limiter = MagicMock()

        result = judge_answer(client, "q", "a", "c", "d", limiter)
        assert result is None

    @patch("src.evaluation.llm_eval.retry_with_backoff")
    def test_judge_answer_handles_null_parsed(self, mock_retry):
        mock_retry.side_effect = lambda fn, **kw: fn()

        from src.evaluation.llm_eval import judge_answer

        client = MagicMock()
        client.beta.chat.completions.parse.return_value.choices[0].message.parsed = None
        limiter = MagicMock()

        result = judge_answer(client, "q", "a", "c", "d", limiter)
        assert result is None


class TestGetDocumentContent:
    @patch("src.evaluation.llm_eval.get_connection")
    def test_found(self, mock_get_conn):
        from src.evaluation.llm_eval import get_document_content

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ("content here",)
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_conn.return_value.__enter__.return_value = mock_conn

        result = get_document_content("doc.md")
        assert result == "content here"

    @patch("src.evaluation.llm_eval.get_connection")
    def test_not_found(self, mock_get_conn):
        from src.evaluation.llm_eval import get_document_content

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_conn.return_value.__enter__.return_value = mock_conn

        result = get_document_content("missing.md")
        assert result is None


class TestStoreRun:
    @patch("src.evaluation.llm_eval.get_connection")
    def test_creates_run(self, mock_get_conn):
        from src.evaluation.llm_eval import store_run

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (42,)
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_conn.return_value.__enter__.return_value = mock_conn

        run_id = store_run("judge-v1", "model-v1", 10, {"key": "val"})
        assert run_id == 42
        mock_get_conn.assert_called_once_with()


class TestStoreResult:
    @patch("src.evaluation.llm_eval.get_connection")
    def test_stores_result(self, mock_get_conn):
        from src.evaluation.llm_eval import store_result
        from src.domain.evaluation import EvaluationScores
        from src.domain.judge import JudgeResult

        scores = EvaluationScores(
            faithfulness_score=5, faithfulness_reasoning="perfect",
            context_relevance_score=4, context_relevance_reasoning="good",
            completeness_score=3, completeness_reasoning="ok",
        )
        judge_result = JudgeResult(scores=scores, input_tokens=100, output_tokens=50, cost=1.5)

        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_conn.return_value.__enter__.return_value = mock_conn

        store_result(1, 100, "doc.md", "context", judge_result)
        mock_get_conn.assert_called_once_with()


class TestEvaluateQuestion:
    @patch("src.evaluation.llm_eval.get_document_content")
    @patch("src.evaluation.llm_eval.judge_answer")
    @patch("src.evaluation.llm_eval.store_result")
    def test_evaluate_question_success(self, mock_store, mock_judge, mock_get_doc):
        from src.evaluation.llm_eval import evaluate_question
        from src.domain.evaluation import EvaluationScores
        from src.domain.judge import JudgeResult

        mock_get_doc.return_value = "expected content"
        mock_scores = EvaluationScores(
            faithfulness_score=4, faithfulness_reasoning="g",
            context_relevance_score=3, context_relevance_reasoning="o",
            completeness_score=5, completeness_reasoning="d",
        )
        mock_judge.return_value = JudgeResult(scores=mock_scores, input_tokens=100, output_tokens=50, cost=1.5)

        assistant = MagicMock()
        assistant.search.return_value = [{"content": "ctx1"}, {"content": "ctx2"}]
        assistant.rag.return_value = {"id": 99, "answer": "the answer"}

        result = evaluate_question(
            {"question": "q?", "document": "doc.md", "_conversation_id": 1},
            run_id=1,
            assistant=assistant,
            judge_client=MagicMock(),
            limiter=MagicMock(),
        )

        assert result is not None
        assert result["question"] == "q?"
        assert result["answer"] == "the answer"
        assert result["faithfulness_score"] == 4
        assert result["judge_input_tokens"] == 100
        assert result["judge_output_tokens"] == 50
        assert result["judge_cost"] == 1.5
        mock_store.assert_called_once()

    @patch("src.evaluation.llm_eval.get_document_content")
    def test_skips_when_doc_missing(self, mock_get_doc):
        from src.evaluation.llm_eval import evaluate_question

        mock_get_doc.return_value = None

        result = evaluate_question(
            {"question": "q?", "document": "missing.md"},
            run_id=1, assistant=MagicMock(),
            judge_client=MagicMock(), limiter=MagicMock(),
        )
        assert result is None

    @patch("src.evaluation.llm_eval.get_document_content")
    def test_skips_when_rag_fails(self, mock_get_doc):
        from src.evaluation.llm_eval import evaluate_question

        mock_get_doc.return_value = "content"
        assistant = MagicMock()
        assistant.rag.side_effect = Exception("RAG error")

        result = evaluate_question(
            {"question": "q?", "document": "doc.md"},
            run_id=1, assistant=assistant,
            judge_client=MagicMock(), limiter=MagicMock(),
        )
        assert result is None


class TestSaveOutputs:
    @patch("src.evaluation.llm_eval.pd.DataFrame.to_csv")
    @patch("src.evaluation.llm_eval.json.dump")
    @patch("builtins.open")
    @patch("src.evaluation.llm_eval.os.makedirs")
    def test_saves_outputs(self, mock_makedirs, mock_open, mock_json, mock_csv):
        from src.evaluation.llm_eval import save_outputs

        save_outputs(
            [{"question": "q1", "faithfulness_score": 4}],
            {"avg_faithfulness": 4.0},
        )

        mock_csv.assert_called_once()
        mock_json.assert_called_once()


class TestMain:
    def test_main_requires_env_vars(self):
        from src.evaluation.llm_eval import main

        with patch("src.evaluation.llm_eval.JUDGE_BASE_URL", None):
            with patch("src.evaluation.llm_eval.JUDGE_MODEL", None):
                with patch("src.evaluation.llm_eval.JUDGE_API_KEY", None):
                    with patch("src.evaluation.llm_eval.load_ground_truth") as m_load:
                        main()

        m_load.assert_not_called()
