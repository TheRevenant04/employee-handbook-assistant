from app.evaluation.llm_eval import main as llm_eval_main
from app.evaluation.search_eval import main as search_eval_main
from app.evaluation.datasets import main as generate_ground_truth

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/run_eval.py [llm|search|ground-truth]")
        sys.exit(1)
    command = sys.argv[1]
    if command == "llm":
        llm_eval_main()
    elif command == "search":
        search_eval_main()
    elif command == "ground-truth":
        generate_ground_truth()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
