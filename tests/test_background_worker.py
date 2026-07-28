import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from background_worker import BackgroundWorker


class TestBackgroundWorker:
    def test_start_worker_creates_thread(self):
        worker = BackgroundWorker()
        worker._start_worker()

        assert hasattr(worker, "_queue")
        assert hasattr(worker, "_worker_thread")
        assert worker._worker_thread.is_alive()
        assert worker._worker_thread.daemon is True

    @patch("background_worker.connect_db")
    def test_submit_executes_task(self, mock_connect_db):
        mock_conn = MagicMock()
        mock_connect_db.return_value = mock_conn

        worker = BackgroundWorker()
        worker._start_worker()
        result = {}

        def task(conn, r):
            r["executed"] = True
            r["conn"] = conn

        worker._submit(task, result)
        worker._queue.join()

        assert result.get("executed") is True
        assert result.get("conn") is mock_conn

    @patch("background_worker.connect_db")
    def test_worker_reconnects_on_task_failure(self, mock_connect_db):
        call_count = 0
        conns = []

        def make_conn():
            nonlocal call_count
            call_count += 1
            conn = MagicMock()
            conn.id = call_count
            conns.append(conn)
            return conn

        mock_connect_db.side_effect = make_conn

        worker = BackgroundWorker()
        worker._start_worker()
        time.sleep(0.1)

        def failing_task(conn):
            raise Exception("task failed")

        def success_task(conn, results):
            results.append(conn.id)

        results = []
        worker._submit(failing_task)
        worker._submit(success_task, results)
        worker._queue.join()

        assert len(conns) >= 2
        assert len(results) == 1

    @patch("background_worker.connect_db")
    def test_worker_handles_task_exception_gracefully(self, mock_connect_db):
        mock_connect_db.return_value = MagicMock()

        worker = BackgroundWorker()
        worker._start_worker()
        time.sleep(0.1)

        def failing_task(conn):
            raise ValueError("bad value")

        results = []

        def success_task(conn, r):
            r.append("ok")

        worker._submit(failing_task)
        worker._submit(success_task, results)
        worker._queue.join()

        assert results == ["ok"]

    @patch("background_worker.connect_db")
    def test_submit_passes_kwargs(self, mock_connect_db):
        mock_connect_db.return_value = MagicMock()

        worker = BackgroundWorker()
        worker._start_worker()
        results = {}

        def task(conn, r, key="default"):
            r["key"] = key

        worker._submit(task, results, key="custom")
        worker._queue.join()

        assert results["key"] == "custom"
