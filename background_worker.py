import logging
import threading
from queue import Queue

from db import get_connection

logger = logging.getLogger(__name__)


class BackgroundWorker:
    def _start_worker(self):
        self._queue = Queue()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def _worker(self):
        while True:
            task = self._queue.get()
            try:
                with get_connection() as conn:
                    task(conn)
            except Exception:
                logger.exception("Background task failed")
            finally:
                self._queue.task_done()

    def _submit(self, fn, *args, **kwargs):
        self._queue.put(lambda conn: fn(conn, *args, **kwargs))
