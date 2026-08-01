import re

_MAX_LENGTH = 100_000


def sanitize_for_llm(text: str, max_length: int = _MAX_LENGTH) -> str:
    if not text:
        return ""
    sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return sanitized[:max_length]


