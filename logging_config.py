import json
import logging
import os
import traceback
from collections import OrderedDict
from datetime import datetime, timezone


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.getenv("LOG_FORMAT", "json")


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
    level = getattr(logging, LOG_LEVEL, logging.INFO)
    handler = logging.StreamHandler()

    if LOG_FORMAT == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)
