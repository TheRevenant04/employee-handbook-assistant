def main():
    import sys
    from src.ui.streamlit_app import main as streamlit_main
    from src.ingestion.pipeline import main as ingest_main
    from src.evaluation.search_eval import main as search_eval_main
    from src.evaluation.llm_eval import main as llm_eval_main
    from src.evaluation.datasets import main as ground_truth_main

    commands = {
        "ui": streamlit_main,
        "ingest": ingest_main,
        "evaluate-search": search_eval_main,
        "evaluate-llm": llm_eval_main,
        "generate-ground-truth": ground_truth_main,
    }

    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print(f"Usage: uv run python -m src.main [{'|'.join(commands)}]")
        sys.exit(1)

    commands[sys.argv[1]]()


if __name__ == "__main__":
    main()


