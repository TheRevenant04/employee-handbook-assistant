import os
from unittest.mock import patch

import pytest


class TestSettings:
    def test_default_values(self):
        from app.core.config import Settings

        s = Settings()
        assert s.github_owner == "madetech"
        assert s.github_repo == "handbook"
        assert s.pg_database == "employee_handbook"
        assert s.pg_host == "localhost"
        assert s.pg_port == 5432
        assert s.vector_dim == 384
        assert s.llm_base_url is None

    def test_custom_values(self):
        from app.core.config import Settings

        s = Settings(pg_host="pg.example.com", pg_port=6432, vector_dim=768)
        assert s.pg_host == "pg.example.com"
        assert s.pg_port == 6432
        assert s.vector_dim == 768

    @patch.dict(os.environ, {
        "PGDATABASE": "test_db",
        "PGUSER": "test_user",
        "PGPASSWORD": "test_pass",
        "PGHOST": "test_host",
        "PGPORT": "9999",
        "VECTOR_DIM": "128",
        "LLM_BASE_URL": "https://llm.example.com",
        "LLM_API_KEY": "sk-test",
        "LLM_MODEL": "gpt-4",
    }, clear=False)
    def test_from_env(self):
        from app.core.config import Settings

        s = Settings.from_env()
        assert s.pg_database == "test_db"
        assert s.pg_user == "test_user"
        assert s.pg_password == "test_pass"
        assert s.pg_host == "test_host"
        assert s.pg_port == 9999
        assert s.vector_dim == 128
        assert s.llm_base_url == "https://llm.example.com"
        assert s.llm_api_key == "sk-test"
        assert s.llm_model == "gpt-4"

    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_defaults(self):
        from app.core.config import Settings

        s = Settings.from_env()
        assert s.pg_database == "employee_handbook"
        assert s.pg_user == "user"
        assert s.pg_port == 5432
        assert s.vector_dim == 384
        assert s.llm_base_url is None
        assert s.llm_api_key is None

    def test_module_level_settings(self):
        from app.core.config import settings

        assert isinstance(settings, object)
        assert hasattr(settings, "pg_database")
