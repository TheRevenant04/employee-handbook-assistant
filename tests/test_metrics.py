import time
from unittest.mock import MagicMock, patch

import pytest


class TestMetricsCollector:
    @patch("src.evaluation.metrics.BackgroundWorker._start_worker")
    @patch("src.evaluation.metrics.get_connection")
    def test_timer_context_manager(self, mock_get_connection, mock_start):
        from src.evaluation.metrics import MetricsCollector

        mock_conn = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_get_connection.return_value = mock_conn
        collector = MetricsCollector()

        with collector.timer() as result:
            time.sleep(0.01)

        assert result["elapsed_ms"] > 0
        assert isinstance(result["elapsed_ms"], float)

    @patch("src.evaluation.metrics.BackgroundWorker._start_worker")
    @patch("src.evaluation.metrics.get_connection")
    def test_record_error_submits_to_worker(self, mock_get_connection, mock_start):
        from src.evaluation.metrics import MetricsCollector

        mock_conn = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_get_connection.return_value = mock_conn
        collector = MetricsCollector()
        collector._submit = MagicMock()

        collector.record_error(
            source="rag.query",
            error_type="ValueError",
            error_message="test error",
            stack_trace="traceback...",
        )

        collector._submit.assert_called_once()

    @patch("src.evaluation.metrics.BackgroundWorker._start_worker")
    @patch("src.evaluation.metrics.get_connection")
    def test_record_error_writes_to_db(self, mock_get_connection, mock_start):
        from src.evaluation.metrics import MetricsCollector

        mock_conn = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_get_connection.return_value = mock_conn
        collector = MetricsCollector()

        collector._record_error(
            mock_conn,
            source="test",
            error_type="Error",
            error_message="msg",
            stack_trace="trace",
        )

        cursor = mock_conn.cursor.return_value.__enter__.return_value
        sql_str = str(cursor.execute.call_args[0][0])
        assert "INSERT" in sql_str
        mock_conn.commit.assert_called()
