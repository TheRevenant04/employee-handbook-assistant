import pytest


class TestVectorStoreBase:
    def test_abstract_methods_defined(self):
        from app.vectorstore.base import VectorStore

        methods = ["connect", "init_schema", "insert", "vector_search", "keyword_search", "hybrid_search"]
        for name in methods:
            assert hasattr(VectorStore, name)
            assert getattr(VectorStore, name).__isabstractmethod__

    def test_cannot_instantiate(self):
        from app.vectorstore.base import VectorStore

        with pytest.raises(TypeError):
            VectorStore()

    def test_concrete_subclass(self):
        from app.vectorstore.base import VectorStore

        class FakeStore(VectorStore):
            def connect(self): pass
            def init_schema(self, table_name, dim): pass
            def insert(self, table_name, rows): pass
            def vector_search(self, query_vector, table_name, num_results): return []
            def keyword_search(self, query_text, table_name, num_results): return []
            def hybrid_search(self, query_text, query_vector, table_name, num_results, alpha): return []

        store = FakeStore()
        assert isinstance(store, VectorStore)
        assert store.vector_search(None, "t", 5) == []
