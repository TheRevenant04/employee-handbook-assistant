import logging
import time
import traceback
import threading
from queue import Queue

from db import connect_db, MAX_RETRIES, RETRY_BASE_DELAY

logger = logging.getLogger(__name__)


class BackgroundWorker:
    def _start_worker(self):
        self._queue = Queue()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def _connect_with_retry(self):
        for attempt in range(MAX_RETRIES):
            try:
                return connect_db()
            except Exception as e:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                if attempt < MAX_RETRIES - 1:
                    logger.warning(
                        "Worker DB connect failed (attempt %d/%d): %s. Retrying in %.1fs...",
                        attempt + 1, MAX_RETRIES, e, delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "Worker DB connect failed after %d attempts: %s",
                        MAX_RETRIES, e,
                    )
                    raise

    def _worker(self):
        thread_conn = None
        while True:
            if thread_conn is None:
                try:
                    thread_conn = self._connect_with_retry()
                except Exception:
                    logger.error(
                        "Worker could not connect, waiting for next task: %s",
                        traceback.format_exc(),
                    )
                    try:
                        task = self._queue.get(timeout=30)
                    except Exception:
                        continue
                    try:
                        self._queue.task_done()
                    except Exception:
                        pass
                    continue

            try:
                task = self._queue.get()
            except Exception:
                logger.error("Worker queue error: %s", traceback.format_exc())
                time.sleep(1)
                continue

            try:
                task(thread_conn)
            except Exception:
                logger.error("Background task failed: %s", traceback.format_exc())
                try:
                    thread_conn.close()
                except Exception:
                    pass
                thread_conn = None
            finally:
                try:
                    self._queue.task_done()
                except Exception:
                    pass

    def _submit(self, fn, *args, **kwargs):
        self._queue.put(lambda conn: fn(conn, *args, **kwargs))
