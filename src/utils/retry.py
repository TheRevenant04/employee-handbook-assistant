import logging
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)


class RateLimiter:
    def __init__(self, max_requests_per_minute: int, logger_name: str | None = None):
        self.max_requests = max_requests_per_minute
        self.window = 60.0
        self.timestamps: list[float] = []
        self.lock = threading.Lock()
        self._log = logging.getLogger(logger_name or __name__)

    def wait(self):
        while True:
            with self.lock:
                now = time.monotonic()
                self.timestamps = [t for t in self.timestamps if t > now - self.window]
                if len(self.timestamps) < self.max_requests:
                    self.timestamps.append(now)
                    return
                sleep_until = self.timestamps[0] + self.window
                sleep_time = sleep_until - now
            if sleep_time > 0:
                self._log.info("Rate limit: sleeping %.1fs", sleep_time)
                time.sleep(sleep_time)


def is_transient_llm_error(e: Exception) -> bool:
    msg = str(e).lower()
    return (
        "429" in str(e)
        or "rate" in msg
        or "500" in str(e) or "502" in str(e) or "503" in str(e)
        or isinstance(e, ConnectionError)
        or "timeout" in msg
    )


def retry_with_backoff(
    fn: Callable[[], T],
    max_retries: int = 5,
    base_delay: float = 2.0,
    is_retryable: Callable[[Exception], bool] | None = None,
    label: str = "operation",
    logger_name: str | None = None,
) -> T:
    if is_retryable is None:
        is_retryable = lambda _: True
    log = logging.getLogger(logger_name or __name__)
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if not is_retryable(e) or attempt >= max_retries - 1:
                log.error("%s failed after %d attempts: %s", label, max_retries, e, exc_info=True)
                raise
            delay = base_delay * (2 ** attempt)
            log.warning(
                "%s failed (attempt %d/%d): %s. Retrying in %.0fs...",
                label, attempt + 1, max_retries, e, delay,
            )
            time.sleep(delay)


def load_ground_truth(path: str) -> list[dict[str, Any]]:
    import pandas as pd

    df = pd.read_csv(path)
    required_columns = {"question", "document"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Ground truth file is missing required columns: {sorted(missing)}")
    df = df.dropna(subset=["question", "document"]).copy()
    df["question"] = df["question"].astype(str).str.strip()
    df["document"] = df["document"].astype(str).str.strip()
    df = df[(df["question"] != "") & (df["document"] != "")]
    return df.to_dict(orient="records")
