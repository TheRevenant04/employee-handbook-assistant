import logging

from psycopg import sql

from db import get_connection
from background_worker import BackgroundWorker

logger = logging.getLogger(__name__)

SCHEMA = "chat"
CONVERSATIONS_TABLE = "conversations"
MESSAGES_TABLE = "messages"
MESSAGE_METRICS_TABLE = "message_metrics"


class ChatStore(BackgroundWorker):
    def __init__(self):
        self._init_schema()
        self._start_worker()

    def _init_schema(self):
        with get_connection(autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {schema}").format(schema=sql.Identifier(SCHEMA)))
                cur.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {schema}.{table} (
                            id BIGSERIAL PRIMARY KEY,
                            title TEXT NOT NULL DEFAULT 'New conversation',
                            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                        );
                        """
                    ).format(schema=sql.Identifier(SCHEMA), table=sql.Identifier(CONVERSATIONS_TABLE))
                )
                cur.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {schema}.{table} (
                            id BIGSERIAL PRIMARY KEY,
                            conversation_id BIGINT NOT NULL REFERENCES {schema}.{conv_table}(id) ON DELETE CASCADE,
                            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                            question TEXT NOT NULL,
                            answer TEXT,
                            rating SMALLINT
                        );
                        """
                    ).format(
                        schema=sql.Identifier(SCHEMA),
                        table=sql.Identifier(MESSAGES_TABLE),
                        conv_table=sql.Identifier(CONVERSATIONS_TABLE),
                    )
                )
                cur.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {schema}.{table} (
                            id BIGSERIAL PRIMARY KEY,
                            message_id BIGINT NOT NULL REFERENCES {schema}.{msg_table}(id) ON DELETE CASCADE,
                            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                            total_latency_ms FLOAT NOT NULL,
                            retrieval_latency_ms FLOAT,
                            llm_latency_ms FLOAT,
                            num_results INT,
                            avg_distance FLOAT,
                            min_distance FLOAT,
                            model TEXT,
                            success BOOLEAN NOT NULL DEFAULT true,
                            input_tokens INT,
                            output_tokens INT,
                            cost NUMERIC(12, 6) NOT NULL DEFAULT 0
                        );
                        """
                    ).format(
                        schema=sql.Identifier(SCHEMA),
                        table=sql.Identifier(MESSAGE_METRICS_TABLE),
                        msg_table=sql.Identifier(MESSAGES_TABLE),
                    )
                )
                cur.execute(
                    sql.SQL("CREATE INDEX IF NOT EXISTS {idx} ON {schema}.{table} (conversation_id, created_at)").format(
                        idx=sql.Identifier("idx_messages_conversation"),
                        schema=sql.Identifier(SCHEMA),
                        table=sql.Identifier(MESSAGES_TABLE),
                    )
                )
                cur.execute(
                    sql.SQL("CREATE INDEX IF NOT EXISTS {idx} ON {schema}.{table} (message_id)").format(
                        idx=sql.Identifier("idx_metrics_message"),
                        schema=sql.Identifier(SCHEMA),
                        table=sql.Identifier(MESSAGE_METRICS_TABLE),
                    )
                )
        logger.info("Chat store schema '%s' initialized", SCHEMA)

    def create_conversation(self, title="New conversation"):
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        """
                        INSERT INTO {schema}.{table} (title)
                        VALUES (%s)
                        RETURNING id
                        """
                    ).format(schema=sql.Identifier(SCHEMA), table=sql.Identifier(CONVERSATIONS_TABLE)),
                    (title,),
                )
                return cur.fetchone()[0]

    def add_message(self, conversation_id, question, answer):
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        """
                        INSERT INTO {schema}.{table} (conversation_id, question, answer)
                        VALUES (%s, %s, %s)
                        RETURNING id
                        """
                    ).format(schema=sql.Identifier(SCHEMA), table=sql.Identifier(MESSAGES_TABLE)),
                    (conversation_id, question, answer),
                )
                message_id = cur.fetchone()[0]
                cur.execute(
                    sql.SQL(
                        """
                        UPDATE {schema}.{conv_table}
                        SET updated_at = now()
                        WHERE id = %s
                        """
                    ).format(schema=sql.Identifier(SCHEMA), conv_table=sql.Identifier(CONVERSATIONS_TABLE)),
                    (conversation_id,),
                )
                return message_id

    def record_metrics(
        self,
        message_id,
        total_latency_ms,
        retrieval_latency_ms=None,
        llm_latency_ms=None,
        num_results=None,
        avg_distance=None,
        min_distance=None,
        model=None,
        success=True,
        input_tokens=None,
        output_tokens=None,
        cost=0.0,
    ):
        self._submit(
            self._record_metrics,
            message_id,
            total_latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
            llm_latency_ms=llm_latency_ms,
            num_results=num_results,
            avg_distance=avg_distance,
            min_distance=min_distance,
            model=model,
            success=success,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
        )

    def _record_metrics(
        self,
        conn,
        message_id,
        total_latency_ms,
        retrieval_latency_ms=None,
        llm_latency_ms=None,
        num_results=None,
        avg_distance=None,
        min_distance=None,
        model=None,
        success=True,
        input_tokens=None,
        output_tokens=None,
        cost=0.0,
    ):
        try:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        """
                        INSERT INTO {schema}.{table}
                            (message_id, total_latency_ms, retrieval_latency_ms,
                             llm_latency_ms, num_results, avg_distance, min_distance,
                             model, success, input_tokens, output_tokens, cost)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """
                    ).format(schema=sql.Identifier(SCHEMA), table=sql.Identifier(MESSAGE_METRICS_TABLE)),
                    (
                        message_id,
                        total_latency_ms,
                        retrieval_latency_ms,
                        llm_latency_ms,
                        num_results,
                        avg_distance,
                        min_distance,
                        model,
                        success,
                        input_tokens,
                        output_tokens,
                        cost,
                    ),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            logger.exception("Failed to record message metrics for message_id=%s", message_id)

    def get_messages(self, conversation_id):
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        """
                        SELECT m.id, m.conversation_id, m.created_at,
                               m.question, m.answer, m.rating,
                               mt.total_latency_ms, mt.retrieval_latency_ms,
                               mt.llm_latency_ms, mt.num_results, mt.avg_distance,
                               mt.min_distance, mt.model, mt.success,
                               mt.input_tokens, mt.output_tokens, mt.cost
                        FROM {schema}.{msg_table} m
                        LEFT JOIN {schema}.{metrics_table} mt ON mt.message_id = m.id
                        WHERE m.conversation_id = %s
                        ORDER BY m.created_at ASC
                        """
                    ).format(
                        schema=sql.Identifier(SCHEMA),
                        msg_table=sql.Identifier(MESSAGES_TABLE),
                        metrics_table=sql.Identifier(MESSAGE_METRICS_TABLE),
                    ),
                    (conversation_id,),
                )
                rows = cur.fetchall()
                return [
                    {
                        "id": r[0],
                        "conversation_id": r[1],
                        "created_at": r[2],
                        "question": r[3],
                        "answer": r[4],
                        "rating": r[5],
                        "total_latency_ms": r[6],
                        "retrieval_latency_ms": r[7],
                        "llm_latency_ms": r[8],
                        "num_results": r[9],
                        "avg_distance": r[10],
                        "min_distance": r[11],
                        "model": r[12],
                        "success": r[13],
                        "input_tokens": r[14],
                        "output_tokens": r[15],
                        "cost": r[16],
                    }
                    for r in rows
                ]

    def rate_message(self, message_id, rating):
        if rating not in (1, -1, None):
            raise ValueError("rating must be 1 (thumbs up), -1 (thumbs down), or None")
        self._submit(self._rate_message, message_id, rating)

    def _rate_message(self, conn, message_id, rating):
        try:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        """
                        UPDATE {schema}.{table}
                        SET rating = %s
                        WHERE id = %s
                        """
                    ).format(schema=sql.Identifier(SCHEMA), table=sql.Identifier(MESSAGES_TABLE)),
                    (rating, message_id),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            logger.exception("Failed to rate message_id=%s with rating=%s", message_id, rating)
