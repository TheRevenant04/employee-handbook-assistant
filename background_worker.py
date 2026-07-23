import logging
import traceback
import threading
from queue import Queue

from db import connect_db

logger = logging.getLogger(__name__)


class BackgroundWorker:
    def _start_worker(self):
        self._queue = Queue()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def _worker(self):
        thread_conn = None
        try:
            thread_conn = connect_db()
            while True:
                task = self._queue.get()
                try:
                    task(thread_conn)
                except Exception:
                    logger.error("Background task failed: %s", traceback.format_exc())
                    try:
                        thread_conn.close()
                    except Exception:
                        pass
                    thread_conn = connect_db()
                finally:
                    self._queue.task_done()
        except Exception:
            logger.error("Worker thread crashed: %s", traceback.format_exc())
        finally:
            if thread_conn:
                try:
                    thread_conn.close()
                except Exception:
                    pass

    def _submit(self, fn, *args, **kwargs):
        self._queue.put(lambda conn: fn(conn, *args, **kwargs))
