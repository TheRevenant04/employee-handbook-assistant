import sys
import os

from dotenv import load_dotenv
from openai import OpenAI

from ingest import fetch_handbook_documents, build_index, load_documents_from_json
from rag import RAG
from embedder import Embedder

def create_assistant():
    load_dotenv()

    #documents = fetch_handbook_documents()
    documents = load_documents_from_json()
    index = build_index(documents)

    return RAG(
        index=index,
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