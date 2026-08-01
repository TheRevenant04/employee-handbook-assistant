import numpy as np
import pytest


class TestDocument:
    def test_default_construction(self):
        from src.domain.document import Document

        doc = Document(path="test.md", content="hello")
        assert doc.path == "test.md"
        assert doc.content == "hello"
        assert doc.embedding is None
        assert doc.metadata == {}

    def test_with_embedding_and_metadata(self):
        from src.domain.document import Document

        emb = np.array([0.1, 0.2, 0.3])
        doc = Document(path="test.md", content="hello", embedding=emb, metadata={"key": "val"})
        assert doc.embedding is emb
        assert doc.metadata == {"key": "val"}


class TestChunk:
    def test_default_construction(self):
        from src.domain.document import Chunk

        chunk = Chunk(document_path="test.md", content="chunk text", chunk_index=0)
        assert chunk.document_path == "test.md"
        assert chunk.content == "chunk text"
        assert chunk.chunk_index == 0
        assert chunk.embedding is None

    def test_with_embedding(self):
        from src.domain.document import Chunk

        emb = np.array([0.1, 0.2])
        chunk = Chunk(document_path="test.md", content="text", chunk_index=1, embedding=emb)
        assert chunk.embedding is emb


class TestQuery:
    def test_default_construction(self):
        from src.domain.query import Query

        q = Query(text="what is leave policy")
        assert q.text == "what is leave policy"
        assert q.rewritten_text is None
        assert q.metadata == {}

    def test_with_rewritten_text(self):
        from src.domain.query import Query

        q = Query(text="leave", rewritten_text="leave policy details")
        assert q.rewritten_text == "leave policy details"


class TestSearchResult:
    def test_default_construction(self):
        from src.domain.query import SearchResult

        r = SearchResult(id=1, path="test.md", content="content")
        assert r.id == 1
        assert r.distance == 0.0
        assert r.score == 0.0
        assert r.score_type == "distance"
        assert r.method == "vector"
        assert r.rerank_score is None

    def test_with_all_fields(self):
        from src.domain.query import SearchResult

        r = SearchResult(id=2, path="doc.md", content="stuff", distance=0.3, score=0.7, score_type="hybrid", method="hybrid_0.5", rerank_score=0.9)
        assert r.score_type == "hybrid"
        assert r.method == "hybrid_0.5"
        assert r.rerank_score == 0.9


class TestAnswer:
    def test_default_construction(self):
        from src.domain.response import Answer

        a = Answer(text="the answer")
        assert a.text == "the answer"
        assert a.message_id is None
        assert a.conversation_id is None
        assert a.sources is None

    def test_with_all_fields(self):
        from src.domain.response import Answer

        a = Answer(text="answer", message_id=42, conversation_id=1, sources=[{"path": "test.md"}])
        assert a.message_id == 42
        assert a.sources == [{"path": "test.md"}]


class TestEvaluationScores:
    def test_default_construction(self):
        from src.domain.response import EvaluationScores

        s = EvaluationScores(
            faithfulness_score=4, faithfulness_reasoning="good",
            context_relevance_score=3, context_relevance_reasoning="ok",
            completeness_score=5, completeness_reasoning="great",
        )
        assert s.faithfulness_score == 4
        assert s.completeness_score == 5
