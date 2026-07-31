import time
import logging
import traceback
from contextlib import contextmanager

from src.retrieval.vectorstore import get_connection
from src.services.background_worker import BackgroundWorker

logger = logging.getLogger(__name__)

INGESTION_METRICS_TABLE = "ingestion_metrics"
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

    def record_ingestion(self, num_documents, ingestion_latency_ms=None, model=None, success=True):
        self._submit(
            self._record_ingestion,
            num_documents,
            ingestion_latency_ms=ingestion_latency_ms,
            model=model,
            success=success,
        )

    def _record_ingestion(self, conn, num_documents, ingestion_latency_ms=None, model=None, success=True):
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {INGESTION_METRICS_TABLE}
                        (num_documents, ingestion_latency_ms, model, success)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (num_documents, ingestion_latency_ms, model, success),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            logger.error("Failed to record ingestion metric: %s", traceback.format_exc())

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


