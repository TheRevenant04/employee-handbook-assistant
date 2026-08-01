import time
import logging
import traceback
from contextlib import contextmanager

from src.retrieval.vectorstore import get_connection
from src.utils.background_worker import BackgroundWorker

logger = logging.getLogger(__name__)

ERROR_LOG_TABLE = "error_log"


class MetricsCollector(BackgroundWorker):
    def __init__(self):
        self._start_worker()

    @contextmanager
    def timer(self):
        start = time.perf_counter()
        result = {"elapsed_ms": 0}
        try:
            yield result
        finally:
            result["elapsed_ms"] = (time.perf_counter() - start) * 1000

    def record_error(self, source, error_type, error_message, stack_trace=None):
        self._submit(
            self._record_error,
            source,
            error_type,
            error_message,
            stack_trace=stack_trace,
        )

    def _record_error(self, conn, source, error_type, error_message, stack_trace=None):
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {ERROR_LOG_TABLE}
                        (source, error_type, error_message, stack_trace)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (source, error_type, error_message, stack_trace),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            logger.error("Failed to record error: %s", traceback.format_exc())


