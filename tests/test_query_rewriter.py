from unittest.mock import MagicMock

import pytest

from src.rag.query_rewriter import QueryRewriter


class TestQueryRewriter:
    def setup_method(self):
        self.mock_llm = MagicMock()

    def test_rewrite_returns_llm_response(self):
        choice = MagicMock()
        choice.message.content = "What is the leave policy?"
        response = MagicMock()
        response.choices = [choice]
        self.mock_llm.chat.completions.create.return_value = response

        rewriter = QueryRewriter(llm_client=self.mock_llm, model="test-model")
        result = rewriter.rewrite("tell me about leave")

        assert result == "What is the leave policy?"

    def test_rewrite_strips_quotes(self):
        choice = MagicMock()
        choice.message.content = '"What is the leave policy?"'
        response = MagicMock()
        response.choices = [choice]
        self.mock_llm.chat.completions.create.return_value = response

        rewriter = QueryRewriter(llm_client=self.mock_llm, model="test-model")
        result = rewriter.rewrite("leave policy?")

        assert result == "What is the leave policy?"

    def test_rewrite_strips_extra_whitespace(self):
        choice = MagicMock()
        choice.message.content = "  What   is   the   policy?  "
        response = MagicMock()
        response.choices = [choice]
        self.mock_llm.chat.completions.create.return_value = response

        rewriter = QueryRewriter(llm_client=self.mock_llm, model="test-model")
        result = rewriter.rewrite("policy?")

        assert result == "What is the policy?"

    def test_rewrite_truncates_long_output(self):
        choice = MagicMock()
        choice.message.content = "x" * 300
        response = MagicMock()
        response.choices = [choice]
        self.mock_llm.chat.completions.create.return_value = response

        rewriter = QueryRewriter(llm_client=self.mock_llm, model="test-model")
        result = rewriter.rewrite("test")

        assert len(result) <= 200

    def test_rewrite_returns_original_on_empty_llm_response(self):
        choice = MagicMock()
        choice.message.content = ""
        response = MagicMock()
        response.choices = [choice]
        self.mock_llm.chat.completions.create.return_value = response

        rewriter = QueryRewriter(llm_client=self.mock_llm, model="test-model")
        result = rewriter.rewrite("my question")

        assert result == "my question"

    def test_rewrite_returns_original_on_exception(self):
        self.mock_llm.chat.completions.create.side_effect = Exception("API error")

        rewriter = QueryRewriter(llm_client=self.mock_llm, model="test-model")
        result = rewriter.rewrite("my question")

        assert result == "my question"

    def test_rewrite_empty_input(self):
        rewriter = QueryRewriter(llm_client=self.mock_llm, model="test-model")
        result = rewriter.rewrite("")

        assert result == ""
        self.mock_llm.chat.completions.create.assert_not_called()

    def test_rewrite_none_input(self):
        rewriter = QueryRewriter(llm_client=self.mock_llm, model="test-model")
        result = rewriter.rewrite(None)

        assert result == ""

    def test_rewrite_truncates_long_input(self):
        choice = MagicMock()
        choice.message.content = "rewritten"
        response = MagicMock()
        response.choices = [choice]
        self.mock_llm.chat.completions.create.return_value = response

        rewriter = QueryRewriter(llm_client=self.mock_llm, model="test-model")
        long_query = "x" * 300
        rewriter.rewrite(long_query)

        call_args = self.mock_llm.chat.completions.create.call_args
        user_content = call_args[1]["messages"][1]["content"]
        assert len(user_content) <= 200

    def test_expand_returns_original_if_no_change(self):
        choice = MagicMock()
        choice.message.content = "same query"
        response = MagicMock()
        response.choices = [choice]
        self.mock_llm.chat.completions.create.return_value = response

        rewriter = QueryRewriter(llm_client=self.mock_llm, model="test-model")
        result = rewriter.expand("same query")

        assert result == ["same query"]

    def test_expand_returns_both_if_changed(self):
        choice = MagicMock()
        choice.message.content = "rewritten query"
        response = MagicMock()
        response.choices = [choice]
        self.mock_llm.chat.completions.create.return_value = response

        rewriter = QueryRewriter(llm_client=self.mock_llm, model="test-model")
        result = rewriter.expand("original query")

        assert len(result) == 2
        assert result[0] == "original query"
        assert result[1] == "rewritten query"

    def test_expand_empty_input(self):
        rewriter = QueryRewriter(llm_client=self.mock_llm, model="test-model")
        result = rewriter.expand("")

        assert result == []
