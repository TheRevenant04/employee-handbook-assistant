from unittest.mock import MagicMock, patch

import pytest


class TestStreamlitApp:
    def _import_module(self):
        with patch("src.ui.streamlit_app.st") as mock_st:
            mock_st.session_state = MagicMock()
            mock_st.session_state.conversation_id = None
            mock_st.session_state.messages = []
            mock_st.session_state.assistant = MagicMock()
            from src.ui.streamlit_app import (
                rate_message, ensure_conversation,
                load_conversation_history,
            )
            return rate_message, ensure_conversation, load_conversation_history

    def test_rate_message(self):
        rate_message, _, _ = self._import_module()

        assistant = MagicMock()
        msg = {"id": 1, "rating": None}
        rate_message(assistant, msg, 1)
        assistant.chat_store.rate_message.assert_called_once_with(1, 1)
        assert msg["rating"] == 1

    def test_rate_message_skips_if_already_rated(self):
        rate_message, _, _ = self._import_module()

        assistant = MagicMock()
        msg = {"id": 1, "rating": -1}
        rate_message(assistant, msg, 1)
        assistant.chat_store.rate_message.assert_not_called()

    def test_rate_message_skips_if_no_id(self):
        rate_message, _, _ = self._import_module()

        assistant = MagicMock()
        msg = {"rating": None}
        rate_message(assistant, msg, 1)
        assistant.chat_store.rate_message.assert_not_called()

    def test_ensure_conversation_reuses_existing(self):
        _, ensure_conversation, _ = self._import_module()

        with patch("src.ui.streamlit_app.st") as mock_st:
            mock_st.session_state.conversation_id = 5
            assistant = MagicMock()
            result = ensure_conversation(assistant, "hello")
            assert result == 5
            assistant.chat_store.create_conversation.assert_not_called()

    def test_ensure_conversation_creates_new(self):
        _, ensure_conversation, _ = self._import_module()

        with patch("src.ui.streamlit_app.st") as mock_st:
            mock_st.session_state.conversation_id = None
            assistant = MagicMock()
            assistant.chat_store.create_conversation.return_value = 10
            result = ensure_conversation(assistant, "my question here")
            assert result == 10
            assert mock_st.session_state.conversation_id == 10
            assistant.chat_store.create_conversation.assert_called_once_with(title="my question here")

    def test_ensure_conversation_truncates_long_title(self):
        _, ensure_conversation, _ = self._import_module()

        with patch("src.ui.streamlit_app.st") as mock_st:
            mock_st.session_state.conversation_id = None
            assistant = MagicMock()
            long = "x" * 100
            ensure_conversation(assistant, long)
            assistant.chat_store.create_conversation.assert_called_once_with(title=long[:80])

    def test_load_conversation_history_skips_without_id(self):
        _, _, load_conversation_history = self._import_module()

        with patch("src.ui.streamlit_app.st") as mock_st:
            mock_st.session_state.conversation_id = None
            mock_st.session_state.messages = []
            assistant = MagicMock()
            load_conversation_history(assistant)
            assistant.chat_store.get_messages.assert_not_called()

    def test_load_conversation_history_skips_if_already_loaded(self):
        _, _, load_conversation_history = self._import_module()

        with patch("src.ui.streamlit_app.st") as mock_st:
            mock_st.session_state.conversation_id = 1
            mock_st.session_state.messages = [{"id": "existing"}]
            assistant = MagicMock()
            load_conversation_history(assistant)
            assistant.chat_store.get_messages.assert_not_called()

