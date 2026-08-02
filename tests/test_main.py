import sys
from unittest.mock import patch

import pytest


class TestMain:
    def test_usage_message(self):
        from src.main import main

        with patch.object(sys, "argv", ["main"]):
            with pytest.raises(SystemExit) as exc:
                main()

        assert exc.value.code == 2

    def test_unknown_command(self):
        from src.main import main

        with patch.object(sys, "argv", ["main", "unknown"]):
            with pytest.raises(SystemExit) as exc:
                main()

        assert exc.value.code == 2

    def test_ui_command(self):
        from src.main import main

        with patch("src.main.run_ui") as mock_fn:
            with patch.object(sys, "argv", ["main", "ui"]):
                main()

        mock_fn.assert_called_once()

    def test_ingest_command(self):
        from src.main import main

        with patch("src.main.run_ingest") as mock_fn:
            with patch.object(sys, "argv", ["main", "ingest"]):
                main()

        mock_fn.assert_called_once()

    def test_evaluate_search_command(self):
        from src.main import main

        with patch("src.main.run_evaluate_search") as mock_fn:
            with patch.object(sys, "argv", ["main", "evaluate-search"]):
                main()

        mock_fn.assert_called_once()

    def test_evaluate_llm_command(self):
        from src.main import main

        with patch("src.main.run_evaluate_llm") as mock_fn:
            with patch.object(sys, "argv", ["main", "evaluate-llm"]):
                main()

        mock_fn.assert_called_once()

    def test_generate_ground_truth_command(self):
        from src.main import main

        with patch("src.main.run_generate_ground_truth") as mock_fn:
            with patch.object(sys, "argv", ["main", "generate-ground-truth"]):
                main()

        mock_fn.assert_called_once()

