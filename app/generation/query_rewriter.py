import logging
import re


logger = logging.getLogger(__name__)


REWRITE_PROMPT = """\
Rewrite the employee's question for document retrieval only when needed.

Rules:
- Preserve the original meaning exactly.
- Prefer terminology likely to appear in an official employee handbook.
- Replace informal wording only when it improves retrieval.
- Do not add new facts, assumptions, or policy categories.
- If the original query is already clear and formal, return it unchanged.
- Return one concise sentence only.

Return ONLY the rewritten query, nothing else.
"""


MAX_QUERY_CHARS = 200
MAX_REWRITTEN_QUERY_CHARS = 200


class QueryRewriter:
    def __init__(self, llm_client, model=None):
        self.llm_client = llm_client
        self.model = model

    def _normalize(self, text: str) -> str:
        text = (text or "").strip()
        text = text.strip('"').strip("'").strip()
        text = re.sub(r"\s+", " ", text)
        return text[:MAX_REWRITTEN_QUERY_CHARS]

    def rewrite(self, query: str) -> str:
        original = self._normalize(query)
        if not original:
            return original

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": REWRITE_PROMPT},
                    {"role": "user", "content": original[:MAX_QUERY_CHARS]},
                ],
                temperature=0.0,
                max_tokens=64,
            )

            rewritten = response.choices[0].message.content
            rewritten = self._normalize(rewritten)

            if not rewritten:
                return original

            logger.info("Query rewritten: %r -> %r", original, rewritten)
            return rewritten

        except Exception:
            logger.exception("Query rewrite failed, using original query")
            return original

    def expand(self, query: str) -> list[str]:
        original = self._normalize(query)
        if not original:
            return []

        rewritten = self.rewrite(original)

        queries = [original]
        if rewritten and rewritten.lower() != original.lower():
            queries.append(rewritten)

        return queries