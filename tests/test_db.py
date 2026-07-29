from unittest.mock import MagicMock, patch, call

import pytest


class TestConnectDb:
    @patch("app.vectorstore.pgvector_store.register_vector")
    @patch("app.vectorstore.pgvector_store.psycopg.connect")
    def test_connects_with_env_vars(self, mock_connect, mock_register):
        from app.vectorstore.pgvector_store import connect_db

        mock_connect.return_value = MagicMock()

        with patch.dict(
            "os.environ",
            {
                "PGDATABASE": "testdb",
                "PGUSER": "testuser",
                "PGPASSWORD": "testpass",
                "PGHOST": "testhost",
                "PGPORT": "5433",
            },
        ):
            connect_db()
            mock_connect.assert_called_once()
            conn_str = mock_connect.call_args[0][0]
            assert "dbname=testdb" in conn_str
            assert "user=testuser" in conn_str
            assert "password=testpass" in conn_str
            assert "host=testhost" in conn_str
            assert "port=5433" in conn_str

    @patch("app.vectorstore.pgvector_store.register_vector")
    @patch("app.vectorstore.pgvector_store.psycopg.connect")
    def test_uses_defaults_when_env_missing(self, mock_connect, mock_register):
        from app.vectorstore.pgvector_store import connect_db

        mock_connect.return_value = MagicMock()

        with patch.dict("os.environ", {}, clear=True):
            connect_db()
            conn_str = mock_connect.call_args[0][0]
            assert "dbname=employee_handbook" in conn_str
            assert "port=5432" in conn_str

    @patch("app.vectorstore.pgvector_store.register_vector")
    @patch("app.core.dependencies.time.sleep")
    @patch("app.vectorstore.pgvector_store.psycopg.connect")
    def test_retries_on_transient_failure(self, mock_connect, mock_sleep, mock_register):
        from app.vectorstore.pgvector_store import connect_db

        mock_connect.side_effect = [Exception("timeout"), MagicMock()]

        result = connect_db()

        assert mock_connect.call_count == 2
        assert result is not None

    @patch("app.vectorstore.pgvector_store.register_vector")
    @patch("app.core.dependencies.time.sleep")
    @patch("app.vectorstore.pgvector_store.psycopg.connect")
    def test_raises_after_max_retries(self, mock_connect, mock_sleep, mock_register):
        from app.vectorstore.pgvector_store import connect_db

        mock_connect.side_effect = Exception("Connection refused")

        with pytest.raises(Exception, match="Connection refused"):
            connect_db()

        assert mock_connect.call_count == 5

    @patch("app.vectorstore.pgvector_store.register_vector")
    @patch("app.core.dependencies.time.sleep")
    @patch("app.vectorstore.pgvector_store.psycopg.connect")
    @patch("app.vectorstore.pgvector_store.RETRY_BASE_DELAY", 2.0)
    def test_exponential_backoff_delays(self, mock_connect, mock_sleep, mock_register):
        from app.vectorstore.pgvector_store import connect_db

        mock_connect.side_effect = [Exception("e1"), Exception("e2"), MagicMock()]

        connect_db()

        delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert delays == [2.0, 4.0]

    @patch("app.vectorstore.pgvector_store.register_vector")
    @patch("app.vectorstore.pgvector_store.psycopg.connect")
    def test_returns_connection_and_registers_vector(self, mock_connect, mock_register):
        from app.vectorstore.pgvector_store import connect_db

        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        result = connect_db()

        assert result is mock_conn
        mock_register.assert_called_once_with(mock_conn)

    @patch("app.vectorstore.pgvector_store.register_vector")
    @patch("app.core.dependencies.time.sleep")
    @patch("app.vectorstore.pgvector_store.psycopg.connect")
    def test_raises_on_invalid_conninfo(self, mock_connect, mock_sleep, mock_register):
        from app.vectorstore.pgvector_store import connect_db

        mock_connect.side_effect = Exception("Invalid connection info")

        with pytest.raises(Exception, match="Invalid connection info"):
            connect_db()


class TestGetConnection:
    @patch("app.vectorstore.pgvector_store.register_vector")
    @patch("app.vectorstore.pgvector_store.connect_db")
    def test_returns_connection_from_connect_db(self, mock_connect_db, mock_register):
        from app.vectorstore import pgvector_store as db

        mock_conn = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_connect_db.return_value = mock_conn

        with db.get_connection() as conn:
            assert conn is mock_conn
        mock_connect_db.assert_called_once_with(autocommit=False)

    @patch("app.vectorstore.pgvector_store.register_vector")
    @patch("app.vectorstore.pgvector_store.connect_db")
    def test_passes_autocommit(self, mock_connect_db, mock_register):
        from app.vectorstore import pgvector_store as db

        mock_conn = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_connect_db.return_value = mock_conn

        with db.get_connection(autocommit=True) as conn:
            assert conn is mock_conn
        mock_connect_db.assert_called_once_with(autocommit=True)


class TestInitDb:
    @patch("app.vectorstore.pgvector_store.get_connection")
    def test_creates_tables_and_indexes(self, mock_get_connection):
        from app.vectorstore.pgvector_store import init_db

        mock_conn = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_get_connection.return_value = mock_conn

        init_db(table_name="test_table", dim=512, index_type="hnsw")

        cursor = mock_conn.cursor.return_value.__enter__.return_value
        assert cursor.execute.call_count >= 4

    @patch("app.vectorstore.pgvector_store.get_connection")
    def test_ivfflat_index_type(self, mock_get_connection):
        from app.vectorstore.pgvector_store import init_db

        mock_conn = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_get_connection.return_value = mock_conn

        init_db(table_name="test_table", dim=384, index_type="ivfflat")

        cursor = mock_conn.cursor.return_value.__enter__.return_value
        calls = [str(c[0][0]) for c in cursor.execute.call_args_list]
        assert any("ivfflat" in c for c in calls)

    def test_invalid_dim_raises(self):
        from app.vectorstore.pgvector_store import init_db

        with pytest.raises(ValueError, match="dim must be a positive integer"):
            init_db(dim=-1)

    def test_non_int_dim_raises(self):
        from app.vectorstore.pgvector_store import init_db

        with pytest.raises(ValueError, match="dim must be a positive integer"):
            init_db(dim="abc")


class TestMigrateAddTsvector:
    @patch("app.vectorstore.pgvector_store.get_connection")
    def test_adds_column_and_index(self, mock_get_connection):
        from app.vectorstore.pgvector_store import migrate_add_tsvector

        mock_conn = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_get_connection.return_value = mock_conn

        migrate_add_tsvector(table_name="test_table")

        cursor = mock_conn.cursor.return_value.__enter__.return_value
        assert cursor.execute.call_count == 2
