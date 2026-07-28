from unittest.mock import MagicMock, patch, call

import pytest


class TestGetPool:
    @patch("db.ConnectionPool")
    @patch("db.register_vector")
    def test_creates_pool_once(self, mock_register, mock_pool_cls):
        import db

        db._pool = None
        mock_pool = MagicMock()
        mock_pool_cls.return_value = mock_pool

        pool1 = db._get_pool()
        pool2 = db._get_pool()

        assert pool1 is pool2
        mock_pool_cls.assert_called_once()

    @patch("db.ConnectionPool")
    @patch("db.register_vector")
    def test_configure_conn_registers_vector(self, mock_register, mock_pool_cls):
        import db

        db._pool = None
        mock_pool_cls.return_value = MagicMock()

        db._get_pool()

        configure_fn = mock_pool_cls.call_args[1]["configure"]
        mock_conn = MagicMock()
        configure_fn(mock_conn)
        mock_register.assert_called_with(mock_conn)

    @patch("db.ConnectionPool")
    @patch("db.register_vector")
    def test_pool_retries_on_failure(self, mock_register, mock_pool_cls):
        import db

        db._pool = None
        mock_pool_cls.side_effect = [Exception("Connection refused"), MagicMock()]

        with patch("db.time.sleep"):
            pool = db._get_pool()

        assert mock_pool_cls.call_count == 2
        assert pool is not None

    @patch("db.ConnectionPool")
    @patch("db.register_vector")
    def test_pool_raises_after_max_retries(self, mock_register, mock_pool_cls):
        import db

        db._pool = None
        mock_pool_cls.side_effect = Exception("Connection refused")

        with patch("db.time.sleep"):
            with pytest.raises(Exception, match="Connection refused"):
                db._get_pool()

        assert mock_pool_cls.call_count == db.MAX_RETRIES

    @patch("db.ConnectionPool")
    @patch("db.register_vector")
    def test_pool_has_health_check(self, mock_register, mock_pool_cls):
        import db

        db._pool = None
        mock_pool_cls.return_value = MagicMock()

        db._get_pool()

        check = mock_pool_cls.call_args[1].get("check")
        assert check is not None


class TestGetConnection:
    @patch("db._get_pool")
    def test_returns_connection_from_pool(self, mock_get_pool):
        import db

        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_pool.connection.return_value = mock_conn
        mock_get_pool.return_value = mock_pool

        result = db.get_connection()
        assert result is mock_conn
        mock_pool.connection.assert_called_once()


class TestConnectDb:
    @patch("db.psycopg.connect")
    def test_connects_with_env_vars(self, mock_connect):
        from db import connect_db

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
            conn_kwargs = mock_connect.call_args[1]
            assert conn_kwargs["dbname"] == "testdb"
            assert conn_kwargs["user"] == "testuser"
            assert conn_kwargs["password"] == "testpass"
            assert conn_kwargs["host"] == "testhost"
            assert conn_kwargs["port"] == 5433

    @patch("db.psycopg.connect")
    def test_uses_defaults_when_env_missing(self, mock_connect):
        from db import connect_db

        mock_connect.return_value = MagicMock()

        with patch.dict("os.environ", {}, clear=True):
            connect_db()
            conn_kwargs = mock_connect.call_args[1]
            assert conn_kwargs["dbname"] == "employee_handbook"
            assert conn_kwargs["port"] == 5432

    @patch("db.time.sleep")
    @patch("db.psycopg.connect")
    def test_retries_on_transient_failure(self, mock_connect, mock_sleep):
        from db import connect_db

        mock_connect.side_effect = [Exception("timeout"), MagicMock()]

        result = connect_db()

        assert mock_connect.call_count == 2
        assert result is not None

    @patch("db.time.sleep")
    @patch("db.psycopg.connect")
    def test_raises_after_max_retries(self, mock_connect, mock_sleep):
        from db import connect_db

        mock_connect.side_effect = Exception("Connection refused")

        with pytest.raises(Exception, match="Connection refused"):
            connect_db()

        assert mock_connect.call_count == 5

    @patch("db.time.sleep")
    @patch("db.psycopg.connect")
    @patch("db.RETRY_BASE_DELAY", 2.0)
    def test_exponential_backoff_delays(self, mock_connect, mock_sleep):
        from db import connect_db

        mock_connect.side_effect = [Exception("e1"), Exception("e2"), MagicMock()]

        connect_db()

        delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert delays == [2.0, 4.0]


class TestInitDb:
    @patch("db.connect_db")
    def test_creates_tables_and_indexes(self, mock_connect_db):
        from db import init_db

        mock_conn = MagicMock()
        mock_connect_db.return_value = mock_conn

        init_db(table_name="test_table", dim=512, index_type="hnsw")

        cursor = mock_conn.cursor.return_value.__enter__.return_value
        assert cursor.execute.call_count >= 4
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch("db.connect_db")
    def test_ivfflat_index_type(self, mock_connect_db):
        from db import init_db

        mock_conn = MagicMock()
        mock_connect_db.return_value = mock_conn

        init_db(table_name="test_table", dim=384, index_type="ivfflat")

        cursor = mock_conn.cursor.return_value.__enter__.return_value
        calls = [str(c[0][0]) for c in cursor.execute.call_args_list]
        assert any("ivfflat" in c for c in calls)

    def test_invalid_dim_raises(self):
        from db import init_db

        with pytest.raises(ValueError, match="dim must be a positive integer"):
            init_db(dim=-1)

    def test_non_int_dim_raises(self):
        from db import init_db

        with pytest.raises(ValueError, match="dim must be a positive integer"):
            init_db(dim="abc")


class TestMigrateAddTsvector:
    @patch("db.connect_db")
    def test_adds_column_and_index(self, mock_connect_db):
        from db import migrate_add_tsvector

        mock_conn = MagicMock()
        mock_connect_db.return_value = mock_conn

        migrate_add_tsvector(table_name="test_table")

        cursor = mock_conn.cursor.return_value.__enter__.return_value
        assert cursor.execute.call_count == 2
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()
