import json
import logging
import sys
from unittest.mock import patch

import pytest


class TestJsonFormatter:
    def test_format_basic_record(self):
        from src.utils.logging import JsonFormatter

        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test_logger", level=logging.INFO,
            pathname="test.py", lineno=42, msg="hello world",
            args=(), exc_info=None,
        )
        record.funcName = "test_func"

        output = formatter.format(record)
        data = json.loads(output)

        assert data["level"] == "INFO"
        assert data["logger"] == "test_logger"
        assert data["module"] == "test"
        assert data["function"] == "test_func"
        assert data["line"] == 42
        assert data["message"] == "hello world"
        assert "timestamp" in data

    def test_format_with_exception(self):
        import traceback
        from src.utils.logging import JsonFormatter

        formatter = JsonFormatter()
        try:
            1 / 0
        except ZeroDivisionError:
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test", level=logging.ERROR,
            pathname="test.py", lineno=10, msg="error occurred",
            args=(), exc_info=exc_info,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert "exception" in data
        assert "ZeroDivisionError" in data["exception"]

    def test_format_with_extra_fields(self):
        from src.utils.logging import JsonFormatter

        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test", level=logging.WARNING,
            pathname="test.py", lineno=5, msg="with extra",
            args=(), exc_info=None,
        )
        record.request_id = "abc-123"

        output = formatter.format(record)
        data = json.loads(output)

        assert data["extra"]["request_id"] == "abc-123"


class TestConfigureLogging:
    def test_configure_json_format(self):
        from src.utils.logging import configure_logging, JsonFormatter

        with patch.object(logging.getLogger(), "hasHandlers", return_value=False):
            with patch.dict("os.environ", {"LOG_FORMAT": "json", "LOG_LEVEL": "DEBUG"}, clear=False):
                configure_logging()

        found = any(isinstance(h.formatter, JsonFormatter) for h in logging.getLogger().handlers)
        assert found

    def test_configure_text_format(self):
        from src.utils.logging import configure_logging

        with patch.object(logging.getLogger(), "hasHandlers", return_value=False):
            with patch.dict("os.environ", {"LOG_FORMAT": "text", "LOG_LEVEL": "WARNING"}, clear=False):
                configure_logging()

        found = any(
            isinstance(h.formatter, logging.Formatter) and not isinstance(h.formatter, type(None))
            for h in logging.getLogger().handlers
        )
        assert found

    def test_idempotent_when_handlers_exist(self):
        from src.utils.logging import configure_logging, JsonFormatter

        with patch.object(logging.getLogger(), "hasHandlers", return_value=True):
            before = len(logging.getLogger().handlers)
            configure_logging()
            assert len(logging.getLogger().handlers) == before

