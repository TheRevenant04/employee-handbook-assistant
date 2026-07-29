import logging
import uuid

import streamlit as st

from app.services.rag_service import create_assistant, get_llm_client, get_reranker, get_query_rewriter
from app.embeddings.provider import Embedder
from app.core.logging import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Employee Handbook Assistant", page_icon="📘")


@st.cache_resource
def load_embedder():
    return Embedder()


@st.cache_resource
def load_llm_client():
    return get_llm_client()


@st.cache_resource
def load_reranker():
    return get_reranker()


@st.cache_resource
def load_query_rewriter():
    return get_query_rewriter()


def init_state():
    if "assistant" not in st.session_state:
        st.session_state.assistant = create_assistant(
            embedder=load_embedder(),
            llm_client=load_llm_client(),
            reranker=load_reranker(),
            query_rewriter=load_query_rewriter(),
        )

    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None

    if "messages" not in st.session_state:
        st.session_state.messages = []


def load_conversation_history(assistant):
    conversation_id = st.session_state.conversation_id
    if conversation_id is None or st.session_state.messages:
        return

    stored_messages = assistant.chat_store.get_messages(conversation_id)
    st.session_state.messages = [
        {
            "question": msg["question"],
            "answer": msg["answer"],
            "id": msg.get("id") or str(uuid.uuid4()),
            "rating": msg.get("rating"),
        }
        for msg in stored_messages
    ]


def ensure_conversation(assistant, first_question):
    if st.session_state.conversation_id is not None:
        return st.session_state.conversation_id

    title = first_question[:80].strip() or "New conversation"
    conversation_id = assistant.chat_store.create_conversation(title=title)
    st.session_state.conversation_id = conversation_id
    return conversation_id


def rate_message(assistant, msg, rating):
    if msg.get("rating") is None and msg.get("id"):
        assistant.chat_store.rate_message(msg["id"], rating)
        msg["rating"] = rating


def render_rating_controls(assistant, msg):
    thumbs_col1, thumbs_col2, _ = st.columns([1, 1, 8])
    already_rated = msg.get("rating") is not None

    with thumbs_col1:
        if st.button("👍", key=f"up_{msg['id']}", disabled=already_rated):
            rate_message(assistant, msg, 1)
            st.rerun()

    with thumbs_col2:
        if st.button("👎", key=f"down_{msg['id']}", disabled=already_rated):
            rate_message(assistant, msg, -1)
            st.rerun()

    if msg.get("rating") == 1:
        st.success("Rated: thumbs up")
    elif msg.get("rating") == -1:
        st.error("Rated: thumbs down")


def render_messages(assistant):
    for msg in st.session_state.messages:
        with st.chat_message("user"):
            st.write(msg["question"])

        with st.chat_message("assistant"):
            st.write(msg["answer"])
            render_rating_controls(assistant, msg)


def main():
    init_state()
    assistant = st.session_state.assistant
    load_conversation_history(assistant)

    st.title("Employee Handbook Assistant")
    render_messages(assistant)

    if user_input := st.chat_input("Ask about the employee handbook"):
        with st.chat_message("user"):
            st.write(user_input)

        conversation_id = ensure_conversation(assistant, user_input)

        with st.spinner("Thinking..."):
            try:
                result = assistant.rag(user_input, conversation_id=conversation_id)
            except Exception as e:
                logger.error("RAG query failed: %s", e, exc_info=True)
                st.error("Something went wrong. Please try again.")
                st.stop()

        answer = result.get("answer", "No answer returned.")
        message_id = result.get("id") or str(uuid.uuid4())

        st.session_state.messages.append(
            {
                "question": user_input,
                "answer": answer,
                "id": message_id,
                "rating": None,
            }
        )
        st.rerun()


if __name__ == "__main__":
    main()