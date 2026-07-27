import os
import logging

from dotenv import load_dotenv
from openai import OpenAI

from rag import RAG
from embedder import Embedder
from metrics import MetricsCollector
from chat_store import ChatStore
from evaluator import Evaluator
from reranker import Reranker


load_dotenv()
logger = logging.getLogger(__name__)


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default

    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off", ""}:
        return False

    raise ValueError(f"Invalid boolean env var {name}={raw!r}")


def get_reranker():
    reranker_enabled = env_bool("RERANKER_ENABLED", default=False)
    reranker_model_path = os.getenv(
        "RERANKER_MODEL_PATH",
        "models/Xenova/ms-marco-MiniLM-L-6-v2",
    )

    if not reranker_enabled:
        logger.info("Reranker disabled via RERANKER_ENABLED=false")
        return None

    if not os.path.exists(reranker_model_path):
        logger.warning(
            "Reranker enabled but model path does not exist: %s",
            reranker_model_path,
        )
        return None

    try:
        reranker = Reranker(reranker_model_path)
        logger.info("Loaded reranker from %s", reranker_model_path)
        return reranker
    except Exception:
        logger.exception(
            "Failed to load reranker from %s, proceeding without reranking",
            reranker_model_path,
        )
        return None


def get_llm_client():
    return OpenAI(
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=os.getenv("LLM_API_KEY"),
    )


def create_assistant(
    embedder=None,
    llm_client=None,
    reranker=None,
):
    embedder = embedder or Embedder()
    llm_client = llm_client or get_llm_client()
    reranker = reranker if reranker is not None else get_reranker()

    chat_store = ChatStore()
    metrics = MetricsCollector()
    evaluator = Evaluator()

    return RAG(
        embedder=embedder,
        llm_client=llm_client,
        chat_store=chat_store,
        model=os.getenv("LLM_MODEL"),
        metrics=metrics,
        evaluator=evaluator,
        reranker=reranker,
    )