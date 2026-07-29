import logging
import threading
from queue import Queue

from db import get_connection

logger = logging.getLogger(__name__)

_STOP_SENTINEL = object()


class BackgroundWorker:
    def _start_worker(self):
        self._queue = Queue()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def stop(self, timeout=5):
        self._queue.put(_STOP_SENTINEL)
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)

    def _worker(self):
        while True:
            try:
                with get_connection() as conn:
                    task = self._queue.get()
                    if task is _STOP_SENTINEL:
                        self._queue.task_done()
                        return
                    try:
                        task(conn)
                    except Exception:
                        logger.exception("Background task failed")
                    finally:
                        self._queue.task_done()
            except Exception:
                logger.exception("Background task failed")

    def _submit(self, fn, *args, **kwargs):
        self._queue.put(lambda conn: fn(conn, *args, **kwargs))
