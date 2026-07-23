import time
import logging
import traceback
from contextlib import contextmanager

from psycopg import sql

from db import connect_db
from background_worker import BackgroundWorker

logger = logging.getLogger(__name__)

SCHEMA = "rag_metrics"

INGESTION_METRICS_TABLE = "ingestion_metrics"
ERROR_LOG_TABLE = "error_log"


class MetricsCollector(BackgroundWorker):
    def __init__(self):
        self._init_schema()
        self._start_worker()

    def _init_schema(self):
        conn = connect_db()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("CREATE SCHEMA IF NOT EXISTS {schema}").format(
                        schema=sql.Identifier(SCHEMA)
                    )
                )

                cur.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {schema}.{table} (
                            id                   BIGSERIAL PRIMARY KEY,
                            timestamp            TIMESTAMPTZ NOT NULL DEFAULT now(),
                            num_documents        INT NOT NULL,
                            ingestion_latency_ms FLOAT,
                            model                TEXT,
                            success              BOOLEAN NOT NULL DEFAULT true
                        );
                        """
                    ).format(
                        schema=sql.Identifier(SCHEMA),
                        table=sql.Identifier(INGESTION_METRICS_TABLE),
                    )
                )

                cur.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {schema}.{table} (
                            id            BIGSERIAL PRIMARY KEY,
                            timestamp     TIMESTAMPTZ NOT NULL DEFAULT now(),
                            source        TEXT NOT NULL,
                            error_type    TEXT NOT NULL,
                            error_message TEXT NOT NULL,
                            stack_trace   TEXT
                        );
                        """
                    ).format(
                        schema=sql.Identifier(SCHEMA),
                        table=sql.Identifier(ERROR_LOG_TABLE),
                    )
                )

            conn.commit()
            logger.info("Metrics schema '%s' initialized", SCHEMA)
        finally:
            conn.close()

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
                    sql.SQL(
                        """
                        INSERT INTO {schema}.{table}
                            (num_documents, ingestion_latency_ms, model, success)
                        VALUES (%s, %s, %s, %s)
                        """
                    ).format(
                        schema=sql.Identifier(SCHEMA),
                        table=sql.Identifier(INGESTION_METRICS_TABLE),
                    ),
                    (num_documents, ingestion_latency_ms, model, success),
                )
            conn.commit()
        except Exception:
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
                    sql.SQL(
                        """
                        INSERT INTO {schema}.{table}
                            (source, error_type, error_message, stack_trace)
                        VALUES (%s, %s, %s, %s)
                        """
                    ).format(
                        schema=sql.Identifier(SCHEMA),
                        table=sql.Identifier(ERROR_LOG_TABLE),
                    ),
                    (source, error_type, error_message, stack_trace),
                )
            conn.commit()
        except Exception:
            logger.error("Failed to record error: %s", traceback.format_exc())
