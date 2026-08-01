from pydantic import BaseModel


class EvaluationScores(BaseModel):
    faithfulness_score: int
    faithfulness_reasoning: str
    context_relevance_score: int
    context_relevance_reasoning: str
    completeness_score: int
    completeness_reasoning: str
