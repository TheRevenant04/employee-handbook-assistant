import sys
import os

from dotenv import load_dotenv
from openai import OpenAI

from rag import RAG
from embedder import Embedder
from db import get_db_connection


def create_assistant():
    load_dotenv()

    db_connection = get_db_connection()

    return RAG(
        db_connection=db_connection,
        embedder= Embedder(),
        llm_client=OpenAI(base_url=os.getenv("LLM_BASE_URL"), api_key=os.getenv("LLM_API_KEY")),
        model=os.getenv("LLM_MODEL"),
    )

if __name__ == "__main__":
    assistant = create_assistant()

    query = "What is the employee handbook?"
    if len(sys.argv) > 1:
        query = sys.argv[1]

    answer = assistant.rag(query)
    print(answer)