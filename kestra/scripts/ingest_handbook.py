"""
Fetches Markdown files from GitHub, embeds locally with ONNX Runtime (light-embed),
and stores in PostgreSQL with pgvector. No PyTorch, no HuggingFace API key required.
"""

import json
import requests
import psycopg2
from psycopg2.extras import execute_values
import os
from typing import List, Dict, Any
from light_embed import TextEmbedding


def get_embedding(model: TextEmbedding, text: str) -> List[float]:
  """Get embedding using ONNX Runtime via light-embed."""
  embedding = model.encode([text])[0]
  return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)


def fetch_git_tree(owner: str, repo: str, branch: str) -> Dict[str, Any]:
  """Fetch the recursive Git tree from GitHub API (public repo, no auth needed)."""
  url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
  response = requests.get(url, timeout=60)
  response.raise_for_status()
  return response.json()


def fetch_raw_content(owner: str, repo: str, branch: str, path: str) -> str:
  """Fetch raw file content from GitHub's CDN (public repo, no auth needed)."""
  url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
  response = requests.get(url, timeout=30)
  response.raise_for_status()
  return response.text


def filter_markdown_files(tree_items: List[Dict[str, Any]]) -> List[str]:
  """Filter tree items to only .md files (blobs, not directories)."""
  return [
      item["path"] for item in tree_items
      if item.get("type") == "blob" and item["path"].endswith(".md")
  ]


def process_and_store(
  model: TextEmbedding,
  markdown_files: List[str],
  owner: str,
  repo: str,
  branch: str,
  db_config: Dict[str, str]
) -> None:
  """Fetch content, embed whole file, and store in PostgreSQL (one row per path)."""
  print(f"Found {len(markdown_files)} markdown files in {owner}/{repo}@{branch}")

  conn = psycopg2.connect(
      host=db_config["POSTGRES_HOST"],
      user=db_config["POSTGRES_USER"],
      password=db_config["POSTGRES_PASSWORD"],
      database=db_config["POSTGRES_DB"]
  )
  conn.autocommit = True
  cur = conn.cursor()

  for doc_idx, md_path in enumerate(markdown_files):
      print(f"Processing [{doc_idx + 1}/{len(markdown_files)}]: {md_path}")

      content = fetch_raw_content(owner, repo, branch, md_path)
      embedding = get_embedding(model, content)

      execute_values(
          cur,
          """
              INSERT INTO handbook_documents (path, content, embedding)
              VALUES %s
              ON CONFLICT (path) DO UPDATE
              SET content = EXCLUDED.content,
                  embedding = EXCLUDED.embedding;
          """,
          [(md_path, content, embedding)],
          template="(%s, %s, %s)"
      )

      if (doc_idx + 1) % 10 == 0:
          print(f"  Processed {doc_idx + 1}/{len(markdown_files)} files...")

  cur.close()
  conn.close()
  print("Embedding and storage complete!")


def main():
  """Main entry point."""
  MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "onnx-models/all-MiniLM-L6-v2-onnx")
  CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/tmp/kestra")

  db_config = {
      "POSTGRES_HOST": os.environ.get("POSTGRES_HOST"),
      "POSTGRES_USER": os.environ.get("POSTGRES_USER"),
      "POSTGRES_PASSWORD": os.environ.get("POSTGRES_PASSWORD"),
      "POSTGRES_DB": os.environ.get("POSTGRES_DB"),
  }

  OWNER = os.environ.get("GITHUB_OWNER", "madetech")
  REPO = os.environ.get("GITHUB_REPO", "handbook")
  BRANCH = os.environ.get("GITHUB_BRANCH", "main")

  print(f"Loading embedding model: {MODEL_NAME}...")

  model = TextEmbedding(MODEL_NAME, cache_folder=CACHE_DIR)

  print(f"Model loaded. Cache directory: {CACHE_DIR}")

  print(f"Fetching Git tree for {OWNER}/{REPO}@{BRANCH}...")
  git_tree_data = fetch_git_tree(OWNER, REPO, BRANCH)

  tree_items = git_tree_data.get("tree", [])
  markdown_files = filter_markdown_files(tree_items)

  process_and_store(
      model=model,
      markdown_files=markdown_files,
      owner=OWNER,
      repo=REPO,
      branch=BRANCH,
      db_config=db_config
  )


if __name__ == "__main__":
  main()
