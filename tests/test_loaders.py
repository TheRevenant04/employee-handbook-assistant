from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest


class TestDownload:
    @patch("app.ingestion.loaders.list_repo_files")
    @patch("app.ingestion.loaders.hf_hub_download")
    @patch("app.ingestion.loaders.shutil.copy2")
    def test_download_auto_detects_onnx(self, mock_copy, mock_download, mock_list):
        mock_list.return_value = ["tokenizer.json", "onnx/model.onnx"]
        mock_download.side_effect = lambda repo_id, filename: f"/cache/{filename}"

        with patch("app.ingestion.loaders.Path.mkdir"):
            from app.ingestion.loaders import download

            download("test/repo", dest="/tmp/models")

        mock_list.assert_called_once_with(repo_id="test/repo")
        assert mock_download.call_count == 2
        mock_download.assert_has_calls([
            call(repo_id="test/repo", filename="tokenizer.json"),
            call(repo_id="test/repo", filename="onnx/model.onnx"),
        ])
        assert mock_copy.call_count == 2

    @patch("app.ingestion.loaders.list_repo_files")
    @patch("app.ingestion.loaders.hf_hub_download")
    @patch("app.ingestion.loaders.shutil.copy2")
    def test_download_with_onnx_filename(self, mock_copy, mock_download, mock_list):
        mock_list.return_value = ["tokenizer.json", "model.onnx"]
        mock_download.side_effect = lambda repo_id, filename: f"/cache/{filename}"

        with patch("app.ingestion.loaders.Path.mkdir"):
            from app.ingestion.loaders import download

            download("test/repo", dest="/tmp/models", onnx_filename="model.onnx")

        mock_download.assert_has_calls([
            call(repo_id="test/repo", filename="tokenizer.json"),
            call(repo_id="test/repo", filename="model.onnx"),
        ])

    @patch("app.ingestion.loaders.list_repo_files")
    def test_raises_when_no_onnx_found(self, mock_list):
        mock_list.return_value = ["tokenizer.json"]

        from app.ingestion.loaders import download

        with pytest.raises(FileNotFoundError, match="No ONNX model found"):
            download("test/repo", dest="/tmp/models")

    @patch("app.ingestion.loaders.list_repo_files")
    @patch("app.ingestion.loaders.hf_hub_download")
    @patch("app.ingestion.loaders.shutil.copy2")
    def test_download_onnx_data(self, mock_copy, mock_download, mock_list):
        mock_list.return_value = ["tokenizer.json", "model.onnx", "model.onnx_data"]
        mock_download.side_effect = lambda repo_id, filename: f"/cache/{filename}"

        with patch("app.ingestion.loaders.Path.mkdir"):
            from app.ingestion.loaders import download

            download("test/repo", dest="/tmp/models")

        mock_download.assert_has_calls([
            call(repo_id="test/repo", filename="tokenizer.json"),
            call(repo_id="test/repo", filename="model.onnx"),
            call(repo_id="test/repo", filename="model.onnx_data"),
        ])
        assert mock_copy.call_count == 3

    @patch("app.ingestion.loaders.list_repo_files")
    @patch("app.ingestion.loaders.hf_hub_download")
    @patch("app.ingestion.loaders.shutil.copy2")
    def test_skips_existing_files(self, mock_copy, mock_download, mock_list):
        mock_list.return_value = ["tokenizer.json", "model.onnx"]
        mock_download.side_effect = lambda repo_id, filename: f"/cache/{filename}"

        def fake_exists(path):
            return True

        with patch("app.ingestion.loaders.Path.mkdir"):
            with patch("app.ingestion.loaders.Path.exists", fake_exists):
                from app.ingestion.loaders import download

                download("test/repo", dest="/tmp/models")

        mock_copy.assert_not_called()
