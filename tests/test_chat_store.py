from unittest.mock import MagicMock, patch, call
import threading

import pytest


class TestChatStoreInit:
    @patch("chat_store.BackgroundWorker._start_worker")
    @patch("chat_store.ChatStore._init_schema")
    def test_init_calls_schema_and_starts_worker(self, mock_init_schema, mock_start):
        from chat_store import ChatStore

        store = ChatStore()
        mock_init_schema.assert_called_once()
        mock_start.assert_called_once()

    @patch("chat_store.BackgroundWorker._start_worker")
    @patch("chat_store.get_connection")
    def test_init_schema_creates_tables(self, mock_get_connection, mock_start):
        from chat_store import ChatStore

        mock_conn = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_get_connection.return_value = mock_conn

        store = ChatStore()

        cursor = mock_conn.cursor.return_value.__enter__.return_value
        assert cursor.execute.call_count >= 5


class TestChatStoreCreateConversation:
    @patch("chat_store.BackgroundWorker._start_worker")
    @patch("chat_store.ChatStore._init_schema")
    @patch("chat_store.get_connection")
    def test_create_conversation_returns_id(self, mock_get_connection, mock_init, mock_start):
        from chat_store import ChatStore

        mock_conn = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_get_connection.return_value = mock_conn
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = [42]

        store = ChatStore()
        result = store.create_conversation("Test Title")

        assert result == 42

    @patch("chat_store.BackgroundWorker._start_worker")
    @patch("chat_store.ChatStore._init_schema")
    @patch("chat_store.get_connection")
    def test_create_conversation_with_default_title(self, mock_get_connection, mock_init, mock_start):
        from chat_store import ChatStore

        mock_conn = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_get_connection.return_value = mock_conn
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = [1]

        store = ChatStore()
        store.create_conversation()

        call_args = cursor.execute.call_args
        assert "New conversation" in str(call_args)


class TestChatStoreAddMessage:
    @patch("chat_store.BackgroundWorker._start_worker")
    @patch("chat_store.ChatStore._init_schema")
    @patch("chat_store.get_connection")
    def test_add_message_returns_id(self, mock_get_connection, mock_init, mock_start):
        from chat_store import ChatStore

        mock_conn = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_get_connection.return_value = mock_conn
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = [100]

        store = ChatStore()
        result = store.add_message(1, "question", "answer")

        assert result == 100
        assert cursor.execute.call_count == 2

    @patch("chat_store.BackgroundWorker._start_worker")
    @patch("chat_store.ChatStore._init_schema")
    @patch("chat_store.get_connection")
    def test_add_message_updates_conversation_timestamp(self, mock_get_connection, mock_init, mock_start):
        from chat_store import ChatStore

        mock_conn = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_get_connection.return_value = mock_conn
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = [1]

        store = ChatStore()
        store.add_message(1, "q", "a")

        second_call = cursor.execute.call_args_list[1]
        sql_str = str(second_call[0][0])
        assert "UPDATE" in sql_str
        assert "updated_at" in sql_str


class TestChatStoreRecordMetrics:
    @patch("chat_store.BackgroundWorker._start_worker")
    @patch("chat_store.ChatStore._init_schema")
    def test_record_metrics_inserts_row(self, mock_init, mock_start):
        from chat_store import ChatStore

        store = ChatStore()
        store._submit = MagicMock()

        store.record_metrics(
            message_id=1,
            total_latency_ms=250.0,
            retrieval_latency_ms=100.0,
            llm_latency_ms=150.0,
            num_results=5,
            avg_distance=0.3,
            min_distance=0.1,
            model="test-model",
            success=True,
            input_tokens=100,
            output_tokens=50,
            cost=0.005,
        )

        store._submit.assert_called_once()


class TestChatStoreGetMessages:
    @patch("chat_store.BackgroundWorker._start_worker")
    @patch("chat_store.ChatStore._init_schema")
    @patch("chat_store.get_connection")
    def test_get_messages_returns_formatted_list(self, mock_get_connection, mock_init, mock_start):
        from chat_store import ChatStore

        mock_conn = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_get_connection.return_value = mock_conn
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = [
            (1, 1, "2024-01-01", "question1", "answer1", 1, 100.0, 50.0, 50.0, 5, 0.3, 0.1, "model", True, 100, 50, 0.005)
        ]

        store = ChatStore()
        messages = store.get_messages(1)

        assert len(messages) == 1
        msg = messages[0]
        assert msg["id"] == 1
        assert msg["question"] == "question1"
        assert msg["answer"] == "answer1"
        assert msg["rating"] == 1

    @patch("chat_store.BackgroundWorker._start_worker")
    @patch("chat_store.ChatStore._init_schema")
    @patch("chat_store.get_connection")
    def test_get_messages_empty(self, mock_get_connection, mock_init, mock_start):
        from chat_store import ChatStore

        mock_conn = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_get_connection.return_value = mock_conn
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = []

        store = ChatStore()
        messages = store.get_messages(999)

        assert messages == []


class TestChatStoreRateMessage:
    @patch("chat_store.BackgroundWorker._start_worker")
    @patch("chat_store.ChatStore._init_schema")
    def test_rate_message_invalid_raises(self, mock_init, mock_start):
        from chat_store import ChatStore

        store = ChatStore()
        with pytest.raises(ValueError, match="rating must be"):
            store.rate_message(1, 5)

    @patch("chat_store.BackgroundWorker._start_worker")
    @patch("chat_store.ChatStore._init_schema")
    def test_rate_message_submits_to_worker(self, mock_init, mock_start):
        from chat_store import ChatStore

        store = ChatStore()
        store._submit = MagicMock()

        store.rate_message(1, 1)
        store._submit.assert_called_once()
