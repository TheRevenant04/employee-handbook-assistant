import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.encode.return_value = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    embedder.encode_batch.return_value = np.array(
        [[0.1, 0.2, 0.3, 0.4]], dtype=np.float32
    )
    return embedder


@pytest.fixture
def mock_llm_client():
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = "This is a test answer."
    usage = MagicMock()
    usage.prompt_tokens = 100
    usage.completion_tokens = 50
    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    client.chat.completions.create.return_value = response
    return client


@pytest.fixture
def mock_chat_store():
    store = MagicMock()
    store.add_message.return_value = 1
    store.create_conversation.return_value = 1
    store.get_messages.return_value = []
    store.record_metrics.return_value = None
    return store


@pytest.fixture
def mock_metrics():
    metrics = MagicMock()

    class FakeTimer:
        def __init__(self):
            self.result = {"elapsed_ms": 100.0}

        def __enter__(self):
            return self.result

        def __exit__(self, *args):
            pass

    metrics.timer = FakeTimer
    return metrics


@pytest.fixture
def mock_reranker():
    reranker = MagicMock()
    reranker.rerank.return_value = [
        {"id": 1, "path": "test.md", "content": "test content", "distance": 0.5, "rerank_score": 0.9}
    ]
    return reranker


@pytest.fixture
def mock_query_rewriter():
    rewriter = MagicMock()
    rewriter.rewrite.return_value = "rewritten query"
    return rewriter


@pytest.fixture
def mock_evaluator():
    return MagicMock()


@pytest.fixture
def sample_search_results():
    return [
        {"id": 1, "path": "policies/leave.md", "content": "Leave policy content", "distance": 0.3},
        {"id": 2, "path": "policies/benefits.md", "content": "Benefits policy content", "distance": 0.5},
    ]


@pytest.fixture
def mock_db_connection():
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn
