import threading
import time
from unittest.mock import MagicMock, patch

import pytest


class TestBackgroundWorker:
    @patch("src.utils.background_worker.get_connection")
    def test_start_worker_creates_thread(self, mock_get_connection):
        from src.utils.background_worker import BackgroundWorker

        mock_get_connection.return_value.__enter__.return_value = MagicMock()
        worker = BackgroundWorker()
        worker._start_worker()

        assert hasattr(worker, "_queue")
        assert hasattr(worker, "_worker_thread")
        assert worker._worker_thread.is_alive()
        assert worker._worker_thread.daemon is True

    @patch("src.utils.background_worker.get_connection")
    def test_submit_executes_task(self, mock_get_connection):
        from src.utils.background_worker import BackgroundWorker

        mock_conn = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_get_connection.return_value = mock_conn

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

    @patch("src.utils.background_worker.get_connection")
    def test_worker_reconnects_on_task_failure(self, mock_get_connection):
        from src.utils.background_worker import BackgroundWorker
        call_count = 0
        conns = []

        def make_conn():
            nonlocal call_count
            call_count += 1
            conn = MagicMock()
            conn.id = call_count
            conn.__enter__.return_value = conn
            conns.append(conn)
            return conn

        mock_get_connection.side_effect = make_conn

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

    @patch("src.utils.background_worker.get_connection")
    def test_worker_handles_task_exception_gracefully(self, mock_get_connection):
        from src.utils.background_worker import BackgroundWorker
        mock_conn = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_get_connection.return_value = mock_conn

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

    @patch("src.utils.background_worker.get_connection")
    def test_submit_passes_kwargs(self, mock_get_connection):
        from src.utils.background_worker import BackgroundWorker
        mock_conn = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_get_connection.return_value = mock_conn

        worker = BackgroundWorker()
        worker._start_worker()
        results = {}

        def task(conn, r, key="default"):
            r["key"] = key

        worker._submit(task, results, key="custom")
        worker._queue.join()

        assert results["key"] == "custom"

    @patch("src.utils.background_worker.get_connection")
    def test_worker_survives_reconnect_failure(self, mock_get_connection):
        from src.utils.background_worker import BackgroundWorker
        call_count = 0

        def make_conn():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                conn = MagicMock(id=1)
                conn.__enter__.return_value = conn
                return conn
            if call_count == 2:
                raise Exception("DB down")
            conn = MagicMock(id=3)
            conn.__enter__.return_value = conn
            return conn

        mock_get_connection.side_effect = make_conn

        worker = BackgroundWorker()
        worker._start_worker()
        time.sleep(0.1)

        def failing_task(conn):
            raise Exception("task failed")

        results = []

        def success_task(conn, r):
            r.append(conn.id)

        worker._submit(failing_task)
        worker._submit(success_task, results)
        worker._queue.join()

        assert results == [3]

    @patch("src.utils.background_worker.get_connection")
    def test_worker_survives_initial_connect_failure(self, mock_get_connection):
        from src.utils.background_worker import BackgroundWorker
        call_count = 0

        def make_conn():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("DB not ready")
            conn = MagicMock(id=call_count)
            conn.__enter__.return_value = conn
            return conn

        mock_get_connection.side_effect = make_conn

        worker = BackgroundWorker()
        worker._start_worker()

        results = []

        def success_task(conn, r):
            r.append(conn.id)

        worker._submit(success_task, results)
        worker._queue.join()

        assert results == [3]

    @patch("src.utils.background_worker.get_connection")
    def test_worker_drains_queue_during_reconnect_backoff(self, mock_get_connection):
        from src.utils.background_worker import BackgroundWorker
        call_count = 0

        def make_conn():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                conn = MagicMock(id=1)
                conn.__enter__.return_value = conn
                return conn
            if call_count == 2:
                raise Exception("DB down")
            conn = MagicMock(id=3)
            conn.__enter__.return_value = conn
            return conn

        mock_get_connection.side_effect = make_conn

        worker = BackgroundWorker()
        worker._start_worker()
        time.sleep(0.1)

        def failing_task(conn):
            raise Exception("fail")

        results = []

        def success_task(conn, r):
            r.append("done")

        worker._submit(failing_task)
        worker._submit(success_task, results)
        worker._queue.join()

        assert results == ["done"]

