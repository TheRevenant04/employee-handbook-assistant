import requests
import json
from pathlib import Path
from minsearch import Index

def fetch_handbook_documents() -> list[dict[str, str]]:
    owner = "madetech"
    repo = "handbook"
    branch = "main"

    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/vnd.github+json",
            "User-Agent": "uv-markdown-fetcher",
        }
    )

    tree_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    tree_response = session.get(tree_url, timeout=30)
    tree_response.raise_for_status()
    tree_data = tree_response.json()

    markdown_paths = [
        item["path"]
        for item in tree_data.get("tree", [])
        if item.get("type") == "blob" and item.get("path", "").endswith(".md")
    ]

    files: list[dict[str, str]] = []

    for path in markdown_paths:
        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
        response = session.get(raw_url, timeout=30)
        response.raise_for_status()

        files.append(
            {
                "path": path,
                "content": response.text,
            }
        )

    return files

def save_documents_to_json(
    documents: list[dict[str, str]],
    output_path: str = "data/employee_handbook_documents.json",
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    return path


def build_index(documents):
    index = Index(
        text_fields=["content"]
    )
    index.fit(documents)
    return index