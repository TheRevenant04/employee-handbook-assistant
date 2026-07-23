import sys
import os

from dotenv import load_dotenv
from openai import OpenAI

from rag import RAG
from embedder import Embedder
from metrics import MetricsCollector
from chat_store import ChatStore


def create_assistant():
    load_dotenv()

    chat_store = ChatStore()
    metrics = MetricsCollector()

    return RAG(
        embedder=Embedder(),
        llm_client=OpenAI(base_url=os.getenv("LLM_BASE_URL"), api_key=os.getenv("LLM_API_KEY")),
        chat_store=chat_store,
        model=os.getenv("LLM_MODEL"),
        metrics=metrics,
    )

if __name__ == "__main__":
    assistant = create_assistant()

    query = "What is the employee handbook?"
    if len(sys.argv) > 1:
        query = sys.argv[1]

    conversation_id = assistant.chat_store.create_conversation(title=query[:80])
    answer = assistant.rag(query, conversation_id=conversation_id)
    print(answer["answer"] if isinstance(answer, dict) else answer)
