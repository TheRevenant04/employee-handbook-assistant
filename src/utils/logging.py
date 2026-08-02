import json
import logging
import os
import traceback
from collections import OrderedDict
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = OrderedDict()
        data["timestamp"] = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        data["level"] = record.levelname
        data["logger"] = record.name
        data["module"] = record.module
        data["function"] = record.funcName
        data["line"] = record.lineno
        data["message"] = record.getMessage()

        if record.exc_info and record.exc_info[0] is not None:
            data["exception"] = "".join(traceback.format_exception(*record.exc_info)).rstrip()

        extras = {k: v for k, v in record.__dict__.items() if k not in logging.LogRecord.__dict__ and not k.startswith("_")}
        if extras:
            data["extra"] = extras

        return json.dumps(data, default=str, ensure_ascii=False)


def configure_logging():
    if logging.getLogger().hasHandlers():
        return
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, log_level, logging.INFO)
    handler = logging.StreamHandler()

    if os.getenv("LOG_FORMAT", "text") == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)


