import os
from dataclasses import dataclass


@dataclass
class Settings:
    github_owner: str = "madetech"
    github_repo: str = "handbook"
    github_branch: str = "main"
    pg_database: str = "employee_handbook"
    pg_user: str = "user"
    pg_password: str = "password"
    pg_host: str = "localhost"
    pg_port: int = 5432
    table_name: str = "employee_handbook"
    vector_dim: int = 384
    model_path: str = "models/Xenova/all-MiniLM-L6-v2"
    reranker_model_path: str = "models/Xenova/ms-marco-MiniLM-L-6-v2"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            github_owner=os.getenv("GITHUB_OWNER", "madetech"),
            github_repo=os.getenv("GITHUB_REPO", "handbook"),
            github_branch=os.getenv("GITHUB_BRANCH", "main"),
            pg_database=os.getenv("PGDATABASE", "employee_handbook"),
            pg_user=os.getenv("PGUSER", "user"),
            pg_password=os.getenv("PGPASSWORD", "password"),
            pg_host=os.getenv("PGHOST", "localhost"),
            pg_port=int(os.getenv("PGPORT", "5432")),
            table_name=os.getenv("TABLE_NAME", "employee_handbook"),
            vector_dim=int(os.getenv("VECTOR_DIM", "384")),
            model_path=os.getenv("MODEL_PATH", "models/Xenova/all-MiniLM-L6-v2"),
            reranker_model_path=os.getenv("RERANKER_MODEL_PATH", "models/Xenova/ms-marco-MiniLM-L-6-v2"),
            llm_base_url=os.getenv("LLM_BASE_URL"),
            llm_api_key=os.getenv("LLM_API_KEY"),
            llm_model=os.getenv("LLM_MODEL"),
        )


settings = Settings.from_env()
