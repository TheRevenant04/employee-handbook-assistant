from pydantic import BaseModel


class Questions(BaseModel):
    questions: list[str]
