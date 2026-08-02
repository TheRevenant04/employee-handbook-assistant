from __future__ import annotations

import argparse
import subprocess
import sys

from src.evaluation.datasets import main as ground_truth_main
from src.evaluation.llm_eval import main as llm_eval_main
from src.evaluation.search_eval import main as search_eval_main
from src.ingestion.pipeline import main as ingest_main


def run_ui() -> int:
    cmd = [sys.executable, "-m", "streamlit", "run", "src/ui/streamlit_app.py"]
    return subprocess.call(cmd)


def run_ingest() -> int:
    ingest_main()
    return 0


def run_evaluate_search() -> int:
    search_eval_main()
    return 0


def run_evaluate_llm() -> int:
    llm_eval_main()
    return 0


def run_generate_ground_truth() -> int:
    ground_truth_main()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.main",
        description="Employee handbook assistant command runner.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    ui_parser = subparsers.add_parser("ui", help="Run the Streamlit UI.")
    ui_parser.set_defaults(func=run_ui)

    ingest_parser = subparsers.add_parser("ingest", help="Run the ingestion pipeline.")
    ingest_parser.set_defaults(func=run_ingest)

    eval_search_parser = subparsers.add_parser("evaluate-search", help="Run search evaluation.")
    eval_search_parser.set_defaults(func=run_evaluate_search)

    eval_llm_parser = subparsers.add_parser("evaluate-llm", help="Run LLM evaluation.")
    eval_llm_parser.set_defaults(func=run_evaluate_llm)

    gt_parser = subparsers.add_parser(
        "generate-ground-truth",
        help="Generate the ground-truth dataset.",
    )
    gt_parser.set_defaults(func=run_generate_ground_truth)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return args.func()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
