from psycopg import sql
import os
import logging
import traceback

from metrics import MetricsCollector
from db import get_connection

logger = logging.getLogger(__name__)

COST_PER_INPUT_TOKEN = float(os.getenv("COST_PER_INPUT_TOKEN", "0"))
COST_PER_OUTPUT_TOKEN = float(os.getenv("COST_PER_OUTPUT_TOKEN", "0"))
TABLE_NAME = os.getenv("TABLE_NAME", "employee_handbook")

INSTRUCTIONS = """
You are an AI assistant that answers employee questions using only the organization's official Employee Handbook provided to you.

Your task is to give accurate, clear, and policy-aligned answers that are fully grounded in the provided documents.

RULES:
1. Answer using only the provided context and retrieved handbook content.
2. Do not use prior knowledge, outside knowledge, assumptions, or common practice unless it is explicitly stated in the provided context.
3. If the answer is not clearly supported by the provided context, say:
   "I'm not able to answer this based on the information currently available in the employee handbook. Please contact HR or your manager for clarification."
4. Do not guess, infer missing policy details, or fabricate answers.
5. Do not combine weak clues from different passages into a confident answer unless the combined answer is explicitly supported.
6. If the question is ambiguous, ask a short clarifying question before answering.
7. If the retrieved context is conflicting or inconsistent, say that the handbook information appears inconsistent and recommend confirmation with HR or the relevant department.
8. Keep answers concise, professional, and easy for employees to understand.
9. When possible, mention the relevant handbook section or policy name.
10. Never present speculation as fact.

ANSWERING PROCESS:
- First, check whether the answer is explicitly present in the provided context.
- If yes, answer directly and clearly.
- If partially supported, state only the supported part and clearly note what is not available.
- If not supported, say you do not know based on the handbook.
- Do not hallucinate.

RESPONSE STYLE:
- Use plain, professional language.
- Be brief but complete.
- Use bullet points when helpful.
- Do not add unnecessary explanations.
- Do not make up examples unless the handbook provides them.

SAFE FALLBACK:
If the answer cannot be found in the provided context, respond exactly with:
"I don't know based on the information available in the handbook."

EXAMPLES:

Example 1:
Employee question: "How many sick leave days do I get each year?"
Behavior:
- If the handbook states the number, provide it clearly.
- If the handbook does not state it, respond:
"I don't know based on the information available in the handbook."

Example 2:
Employee question: "Can I carry forward unused vacation days?"
Behavior:
- Answer only if the policy is explicitly stated in the context.
- If the context is silent or unclear, say you do not know.

Example 3:
Employee question: "What happens if I work on a public holiday?"
Behavior:
- If the retrieved context contains the rule, summarize it accurately.
- If not, do not guess based on common HR practice.

IMPORTANT CONSTRAINT:
Your answers must be traceable to the provided handbook context. If a statement is not supported by the context, do not include it.
"""

PROMPT_TEMPLATE = '''
QUESTION: {question}

CONTEXT:
{context}
'''.strip()

class RAG:
    def __init__(
        self,
        embedder,
        llm_client,
        chat_store,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        model=None,
        metrics=None,
        cost_per_input_token=COST_PER_INPUT_TOKEN,
        cost_per_output_token=COST_PER_OUTPUT_TOKEN,
        evaluator=None,
        reranker=None,
    ):
        self.embedder = embedder
        self.llm_client = llm_client
        self.chat_store = chat_store
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model or os.getenv("LLM_MODEL")
        self.metrics = metrics or MetricsCollector()
        self.cost_per_input_token = cost_per_input_token
        self.cost_per_output_token = cost_per_output_token
        self.evaluator = evaluator
        self.reranker = reranker

    def vector_search(self, query_text, num_results=5):
        query_vector = self.embedder.encode(query_text, normalize=True)

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        """
                        SELECT id, path, content, embedding <=> %s AS distance
                        FROM {table}
                        ORDER BY embedding <=> %s
                        LIMIT %s
                        """
                    ).format(
                        table=sql.Identifier(TABLE_NAME),
                    ),
                    (query_vector, query_vector, num_results),
                )
                rows = cur.fetchall()
            return [
                {
                    "id": row[0],
                    "path": row[1],
                    "content": row[2],
                    "distance": float(row[3]),
                }
                for row in rows
            ]

    def keyword_search(self, query_text, num_results=5):
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        """
                        SELECT id, path, content,
                               ts_rank_cd(content_tsv, plainto_tsquery('english', %s)) AS rank
                        FROM {table}
                        WHERE content_tsv @@ plainto_tsquery('english', %s)
                        ORDER BY rank DESC
                        LIMIT %s
                        """
                    ).format(
                        table=sql.Identifier(TABLE_NAME),
                    ),
                    (query_text, query_text, num_results),
                )
                rows = cur.fetchall()
            return [
                {
                    "id": row[0],
                    "path": row[1],
                    "content": row[2],
                    "rank": float(row[3]),
                }
                for row in rows
            ]

    def hybrid_search(self, query_text, num_results=5, alpha=0.5):
        query_vector = self.embedder.encode(query_text, normalize=True)
        fetch_k = num_results * 3

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        """
                        WITH vector_results AS (
                            SELECT id, path, content,
                                   embedding <=> %s AS v_distance,
                                   ROW_NUMBER() OVER (ORDER BY embedding <=> %s) AS v_rank
                            FROM {table}
                            ORDER BY embedding <=> %s
                            LIMIT %s
                        ),
                        keyword_results AS (
                            SELECT id, path, content,
                                   ts_rank_cd(content_tsv, plainto_tsquery('english', %s)) AS k_score,
                                   ROW_NUMBER() OVER (
                                       ORDER BY ts_rank_cd(content_tsv, plainto_tsquery('english', %s)) DESC
                                   ) AS k_rank
                            FROM {table}
                            WHERE content_tsv @@ plainto_tsquery('english', %s)
                            LIMIT %s
                        ),
                        combined AS (
                            SELECT
                                COALESCE(v.id, k.id) AS id,
                                COALESCE(v.path, k.path) AS path,
                                COALESCE(v.content, k.content) AS content,
                                COALESCE(1.0 / (1.0 + v.v_distance), 0) AS v_score,
                                COALESCE(k.k_score, 0) AS k_score,
                                COALESCE(v.v_rank, %s) AS v_rank,
                                COALESCE(k.k_rank, %s) AS k_rank
                            FROM vector_results v
                            FULL OUTER JOIN keyword_results k ON v.id = k.id
                        )
                        SELECT id, path, content,
                               %s * (1.0 / (1.0 + v_rank)) + (1.0 - %s) * (1.0 / (1.0 + k_rank)) AS score
                        FROM combined
                        ORDER BY score DESC
                        LIMIT %s
                        """
                    ).format(
                        table=sql.Identifier(TABLE_NAME),
                    ),
                    (
                        query_vector, query_vector, query_vector, fetch_k,
                        query_text, query_text, query_text, fetch_k,
                        fetch_k, fetch_k,
                        alpha, alpha,
                        num_results,
                    ),
                )
                rows = cur.fetchall()
            return [
                {
                    "id": row[0],
                    "path": row[1],
                    "content": row[2],
                    "distance": -float(row[3]),
                }
                for row in rows
            ]

    def search(self, query_text, num_results=5):
        results = self.hybrid_search(query_text, num_results)
        if self.reranker and results:
            results = self.reranker.rerank(query_text, results, top_k=num_results)
        return results

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            section = doc.get("content", "")
            lines.append(section)
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(question=query, context=context)

    def llm(self, prompt):
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.instructions},
                {"role": "user", "content": prompt},
            ],
        )
        usage = response.usage
        input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        cost = (
            input_tokens * self.cost_per_input_token
            + output_tokens * self.cost_per_output_token
        )
        text = response.choices[0].message.content if response.choices else ""
        return {
            "text": text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
        }

    def rag(self, query, conversation_id, num_results=5):
        with self.metrics.timer() as total_timer:
            retrieval_latency_ms = None
            llm_latency_ms = None
            num_results_returned = None
            avg_distance = None
            min_distance = None
            input_tokens = None
            output_tokens = None
            cost = 0.0
            answer = ""
            retrieved_context = ""
            success = True

            try:
                with self.metrics.timer() as search_timer:
                    search_results = self.search(query, num_results=num_results)
                retrieval_latency_ms = search_timer["elapsed_ms"]

                retrieved_context = self.build_context(search_results)

                num_results_returned = len(search_results)
                if search_results:
                    distances = [r["distance"] for r in search_results]
                    avg_distance = sum(distances) / len(distances)
                    min_distance = min(distances)

                prompt = self.build_prompt(query, search_results)

                with self.metrics.timer() as llm_timer:
                    llm_result = self.llm(prompt)
                llm_latency_ms = llm_timer["elapsed_ms"]

                answer = llm_result["text"]
                input_tokens = llm_result["input_tokens"]
                output_tokens = llm_result["output_tokens"]
                cost = llm_result["cost"]

            except Exception as e:
                success = False
                self.metrics.record_error(
                    source="rag.query",
                    error_type=type(e).__name__,
                    error_message=str(e),
                    stack_trace=traceback.format_exc(),
                )
                raise
            finally:
                try:
                    message_id = self.chat_store.add_message(
                        conversation_id=conversation_id,
                        question=query,
                        answer=answer,
                    )
                    self.chat_store.record_metrics(
                        message_id=message_id,
                        total_latency_ms=total_timer["elapsed_ms"],
                        retrieval_latency_ms=retrieval_latency_ms,
                        llm_latency_ms=llm_latency_ms,
                        num_results=num_results_returned,
                        avg_distance=avg_distance,
                        min_distance=min_distance,
                        model=self.model,
                        success=success,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cost=cost,
                    )
                except Exception:
                    logger.error("Failed to persist chat message: %s", traceback.format_exc())
                    message_id = None

        if self.evaluator and message_id and success:
            self.evaluator.evaluate(
                message_id=message_id,
                question=query,
                answer=answer,
                retrieved_context=retrieved_context,
            )

        return {
            "id": message_id,
            "answer": answer,
            "query": query,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "total_latency_ms": total_timer["elapsed_ms"],
            "retrieval_latency_ms": retrieval_latency_ms,
            "llm_latency_ms": llm_latency_ms,
            "num_results": num_results_returned,
            "model": self.model,
        }
