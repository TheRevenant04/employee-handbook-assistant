from pydantic import BaseModel

from src.domain.evaluation import EvaluationScores


class JudgeResult(BaseModel):
    scores: EvaluationScores
    input_tokens: int
    output_tokens: int
    cost: float
