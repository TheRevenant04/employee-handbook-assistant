from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest


class TestDownload:
    @patch("scripts.download_models.list_repo_files")
    @patch("scripts.download_models.hf_hub_download")
    @patch("scripts.download_models.shutil.copy2")
    def test_download_auto_detects_onnx(self, mock_copy, mock_download, mock_list):
        mock_list.return_value = ["tokenizer.json", "onnx/model.onnx"]
        mock_download.side_effect = lambda repo_id, filename: f"/cache/{filename}"

        with patch("scripts.download_models.Path.mkdir"):
            from scripts.download_models import download

            download("test/repo", dest="/tmp/models")

        mock_list.assert_called_once_with(repo_id="test/repo")
        assert mock_download.call_count == 2
        mock_download.assert_has_calls([
            call(repo_id="test/repo", filename="tokenizer.json"),
            call(repo_id="test/repo", filename="onnx/model.onnx"),
        ])
        assert mock_copy.call_count == 2

    @patch("scripts.download_models.list_repo_files")
    @patch("scripts.download_models.hf_hub_download")
    @patch("scripts.download_models.shutil.copy2")
    def test_download_with_onnx_filename(self, mock_copy, mock_download, mock_list):
        mock_list.return_value = ["tokenizer.json", "model.onnx"]
        mock_download.side_effect = lambda repo_id, filename: f"/cache/{filename}"

        with patch("scripts.download_models.Path.mkdir"):
            from scripts.download_models import download

            download("test/repo", dest="/tmp/models", onnx_filename="model.onnx")

        mock_download.assert_has_calls([
            call(repo_id="test/repo", filename="tokenizer.json"),
            call(repo_id="test/repo", filename="model.onnx"),
        ])

    @patch("scripts.download_models.list_repo_files")
    def test_raises_when_no_onnx_found(self, mock_list):
        mock_list.return_value = ["tokenizer.json"]

        from scripts.download_models import download

        with pytest.raises(FileNotFoundError, match="No ONNX model found"):
            download("test/repo", dest="/tmp/models")

    @patch("scripts.download_models.list_repo_files")
    @patch("scripts.download_models.hf_hub_download")
    @patch("scripts.download_models.shutil.copy2")
    def test_download_onnx_data(self, mock_copy, mock_download, mock_list):
        mock_list.return_value = ["tokenizer.json", "model.onnx", "model.onnx_data"]
        mock_download.side_effect = lambda repo_id, filename: f"/cache/{filename}"

        with patch("scripts.download_models.Path.mkdir"):
            from scripts.download_models import download

            download("test/repo", dest="/tmp/models")

        mock_download.assert_has_calls([
            call(repo_id="test/repo", filename="tokenizer.json"),
            call(repo_id="test/repo", filename="model.onnx"),
            call(repo_id="test/repo", filename="model.onnx_data"),
        ])
        assert mock_copy.call_count == 3

    @patch("scripts.download_models.list_repo_files")
    @patch("scripts.download_models.hf_hub_download")
    @patch("scripts.download_models.shutil.copy2")
    def test_skips_existing_files(self, mock_copy, mock_download, mock_list):
        mock_list.return_value = ["tokenizer.json", "model.onnx"]
        mock_download.side_effect = lambda repo_id, filename: f"/cache/{filename}"

        def fake_exists(path):
            return True

        with patch("scripts.download_models.Path.mkdir"):
            with patch("scripts.download_models.Path.exists", fake_exists):
                from scripts.download_models import download

                download("test/repo", dest="/tmp/models")

        mock_copy.assert_not_called()
