from unittest.mock import MagicMock, patch

import pytest


class TestRateLimiter:
    def test_allows_within_rate(self):
        from evaluator import RateLimiter

        limiter = RateLimiter(max_requests_per_minute=10)
        limiter.wait()
        limiter.wait()
        limiter.wait()

        assert len(limiter.timestamps) == 3

    def test_blocks_over_rate(self):
        from evaluator import RateLimiter

        limiter = RateLimiter(max_requests_per_minute=2)
        limiter.wait()
        limiter.wait()

        assert len(limiter.timestamps) == 2


class TestEvaluator:
    @patch("db.init_llm_evaluation_schema")
    @patch.dict(
        "os.environ",
        {
            "JUDGE_BASE_URL": "http://localhost:11434/v1",
            "JUDGE_MODEL": "test-judge",
            "JUDGE_API_KEY": "test-key",
        },
    )
    def test_init_enables_when_env_set(self, mock_init_schema):
        from evaluator import Evaluator

        ev = Evaluator()
        assert ev._enabled is True

    @patch.dict("os.environ", {}, clear=True)
    def test_init_disables_when_env_missing(self):
        import evaluator
        evaluator.JUDGE_BASE_URL = None
        evaluator.JUDGE_MODEL = None
        evaluator.JUDGE_API_KEY = None

        from evaluator import Evaluator

        ev = Evaluator()
        assert ev._enabled is False

    @patch("db.init_llm_evaluation_schema")
    @patch.dict(
        "os.environ",
        {
            "JUDGE_BASE_URL": "http://localhost",
            "JUDGE_MODEL": "test-judge",
            "JUDGE_API_KEY": "test-key",
        },
    )
    def test_evaluate_skips_when_disabled(self, mock_init_schema):
        import evaluator
        evaluator.JUDGE_BASE_URL = None
        evaluator.JUDGE_MODEL = None
        evaluator.JUDGE_API_KEY = None

        from evaluator import Evaluator

        ev = Evaluator()
        ev._enabled = False

        ev.evaluate(
            message_id=1,
            question="q",
            answer="a",
            retrieved_context="ctx",
        )

    @patch("db.init_llm_evaluation_schema")
    @patch.dict(
        "os.environ",
        {
            "JUDGE_BASE_URL": "http://localhost",
            "JUDGE_MODEL": "test-judge",
            "JUDGE_API_KEY": "test-key",
            "EVAL_SAMPLE_RATE": "0",
        },
    )
    def test_evaluate_skips_when_sample_rate_zero(self, mock_init_schema):
        from evaluator import Evaluator

        ev = Evaluator()
        ev._enabled = True

        with patch.object(ev, "_run_evaluation") as mock_run:
            ev.evaluate(
                message_id=1,
                question="q",
                answer="a",
                retrieved_context="ctx",
            )
            mock_run.assert_not_called()

    @patch("db.init_llm_evaluation_schema")
    @patch.dict(
        "os.environ",
        {
            "JUDGE_BASE_URL": "http://localhost",
            "JUDGE_MODEL": "test-judge",
            "JUDGE_API_KEY": "test-key",
        },
    )
    def test_evaluate_skips_already_evaluated(self, mock_init_schema):
        from evaluator import Evaluator

        ev = Evaluator()
        ev._enabled = True

        with patch.object(ev, "_already_evaluated", return_value=True) as mock_check:
            with patch.object(ev, "_run_evaluation") as mock_run:
                ev.evaluate(
                    message_id=1,
                    question="q",
                    answer="a",
                    retrieved_context="ctx",
                )
                mock_run.assert_not_called()

    @patch("db.init_llm_evaluation_schema")
    @patch.dict(
        "os.environ",
        {
            "JUDGE_BASE_URL": "http://localhost",
            "JUDGE_MODEL": "test-judge",
            "JUDGE_API_KEY": "test-key",
        },
    )
    def test_judge_returns_scores(self, mock_init_schema):
        from evaluator import Evaluator, EvaluationScores

        ev = Evaluator()

        parsed = EvaluationScores(
            faithfulness_score=4,
            faithfulness_reasoning="mostly supported",
            context_relevance_score=5,
            context_relevance_reasoning="highly relevant",
            completeness_score=3,
            completeness_reasoning="partial",
        )

        choice = MagicMock()
        choice.message.parsed = parsed
        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 50
        response = MagicMock()
        response.choices = [choice]
        response.usage = usage

        ev._client = MagicMock()
        ev._client.beta.chat.completions.parse.return_value = response
        ev._limiter = MagicMock()

        result = ev._judge("question", "answer", "context")

        assert result is not None
        assert result.scores.faithfulness_score == 4
        assert result.scores.context_relevance_score == 5
        assert result.scores.completeness_score == 3
        assert result.input_tokens == 100
        assert result.output_tokens == 50

    @patch("db.init_llm_evaluation_schema")
    @patch.dict(
        "os.environ",
        {
            "JUDGE_BASE_URL": "http://localhost",
            "JUDGE_MODEL": "test-judge",
            "JUDGE_API_KEY": "test-key",
        },
    )
    def test_judge_returns_none_on_failure(self, mock_init_schema):
        from evaluator import Evaluator

        ev = Evaluator()
        ev._client = MagicMock()
        ev._client.beta.chat.completions.parse.side_effect = Exception("API error")
        ev._limiter = MagicMock()

        result = ev._judge("question", "answer", "context")
        assert result is None

    @patch("db.init_llm_evaluation_schema")
    @patch("evaluator.JUDGE_COST_PER_OUTPUT_TOKEN", 0.001)
    @patch("evaluator.JUDGE_COST_PER_INPUT_TOKEN", 0.001)
    @patch("evaluator.JUDGE_MODEL", "test-judge")
    def test_store_result_inserts(self, mock_init_schema):
        from evaluator import Evaluator, JudgeResult, EvaluationScores

        ev = Evaluator.__new__(Evaluator)
        ev._enabled = True

        judge_result = JudgeResult(
            scores=EvaluationScores(
                faithfulness_score=4,
                faithfulness_reasoning="reason",
                context_relevance_score=5,
                context_relevance_reasoning="reason",
                completeness_score=3,
                completeness_reasoning="reason",
            ),
            input_tokens=100,
            output_tokens=50,
            cost=0.01,
        )

        mock_cursor = MagicMock()
        cursor_cm = MagicMock()
        cursor_cm.__enter__ = MagicMock(return_value=mock_cursor)
        cursor_cm.__exit__ = MagicMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = cursor_cm

        conn_cm = MagicMock()
        conn_cm.__enter__ = MagicMock(return_value=mock_conn)
        conn_cm.__exit__ = MagicMock(return_value=False)

        with patch("evaluator.get_connection", return_value=conn_cm):
            ev._store_result(run_id=1, message_id=1, retrieved_context="ctx", judge_result=judge_result)

        mock_cursor.execute.assert_called_once()
        mock_conn.commit.assert_called_once()
