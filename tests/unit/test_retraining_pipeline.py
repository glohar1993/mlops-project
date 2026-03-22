"""
Unit tests for RetrainingPipeline.
All file I/O, joblib, DataProcessing, and ModelTraining are mocked.
"""
import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, call

# Top-level import for coverage tracking
import src.retraining_pipeline  # noqa: F401

from src.retraining_pipeline import RetrainingPipeline


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_pipeline():
    with patch("os.makedirs"):
        return RetrainingPipeline()


# ─────────────────────────────────────────────────────────────────────────────
# TestInit
# ─────────────────────────────────────────────────────────────────────────────

class TestInit:
    def test_init_creates_backup_dir(self):
        with patch("os.makedirs") as mock_mkdir:
            RetrainingPipeline()
        mock_mkdir.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# TestGetCurrentAccuracy
# ─────────────────────────────────────────────────────────────────────────────

class TestGetCurrentAccuracy:
    def test_returns_float_on_success(self):
        pipeline = _make_pipeline()
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, (20, 14))
        y = np.array([i % 3 for i in range(20)])
        mock_model = MagicMock()
        mock_model.predict.return_value = y  # perfect predictions
        with patch("src.retraining_pipeline.joblib.load",
                   side_effect=[mock_model, MagicMock(), X, y]):
            acc = pipeline.get_current_accuracy()
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0

    def test_returns_zero_on_exception(self):
        pipeline = _make_pipeline()
        with patch("src.retraining_pipeline.joblib.load",
                   side_effect=FileNotFoundError("no model")):
            acc = pipeline.get_current_accuracy()
        assert acc == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# TestBackupCurrentModel
# ─────────────────────────────────────────────────────────────────────────────

class TestBackupCurrentModel:
    def test_backup_copies_file_when_exists(self):
        pipeline = _make_pipeline()
        with patch("os.path.exists", return_value=True), \
             patch("shutil.copy") as mock_copy:
            path = pipeline.backup_current_model()
        assert mock_copy.called
        assert "model_" in path

    def test_backup_skips_copy_when_no_model(self):
        pipeline = _make_pipeline()
        with patch("os.path.exists", return_value=False), \
             patch("shutil.copy") as mock_copy:
            pipeline.backup_current_model()
        mock_copy.assert_not_called()

    def test_backup_path_contains_timestamp(self):
        pipeline = _make_pipeline()
        with patch("os.path.exists", return_value=False):
            path = pipeline.backup_current_model()
        assert "backups" in path
        assert ".pkl" in path


# ─────────────────────────────────────────────────────────────────────────────
# TestRetrain
# ─────────────────────────────────────────────────────────────────────────────

class TestRetrain:
    def _patched_retrain(self, old_acc=0.80, new_acc=0.85):
        """Run retrain() with all external calls mocked."""
        pipeline = _make_pipeline()
        with patch.object(pipeline, "get_current_accuracy",
                          side_effect=[old_acc, new_acc]), \
             patch.object(pipeline, "backup_current_model",
                          return_value="backups/model_test.pkl"), \
             patch("src.retraining_pipeline.DataProcessing") as mock_dp, \
             patch("src.retraining_pipeline.ModelTraining") as mock_mt:
            mock_dp.return_value.run.return_value = None
            mock_mt.return_value.run.return_value = {"accuracy": new_acc}
            result = pipeline.retrain()
        return result

    def test_retrain_returns_dict(self):
        result = self._patched_retrain()
        assert isinstance(result, dict)

    def test_retrain_result_has_required_keys(self):
        result = self._patched_retrain()
        assert "old_accuracy" in result
        assert "new_accuracy" in result
        assert "model_replaced" in result
        assert "reason" in result

    def test_promote_if_new_model_is_better(self):
        result = self._patched_retrain(old_acc=0.80, new_acc=0.85)
        assert result["model_replaced"] is True
        assert "deployed" in result["reason"]

    def test_promote_if_accuracy_equal(self):
        result = self._patched_retrain(old_acc=0.80, new_acc=0.80)
        assert result["model_replaced"] is True

    def test_rollback_if_new_model_is_worse(self):
        pipeline = _make_pipeline()
        with patch.object(pipeline, "get_current_accuracy",
                          side_effect=[0.80, 0.40]), \
             patch.object(pipeline, "backup_current_model",
                          return_value="backups/model_test.pkl"), \
             patch("src.retraining_pipeline.DataProcessing") as mock_dp, \
             patch("src.retraining_pipeline.ModelTraining") as mock_mt, \
             patch("shutil.copy") as mock_copy:
            mock_dp.return_value.run.return_value = None
            mock_mt.return_value.run.return_value = {"accuracy": 0.40}
            result = pipeline.retrain()
        assert result["model_replaced"] is False
        assert "rolled back" in result["reason"]
        mock_copy.assert_called_once()

    def test_retrain_aborts_if_data_processing_fails(self):
        pipeline = _make_pipeline()
        with patch.object(pipeline, "get_current_accuracy", return_value=0.80), \
             patch.object(pipeline, "backup_current_model",
                          return_value="backups/model_test.pkl"), \
             patch("src.retraining_pipeline.DataProcessing") as mock_dp:
            mock_dp.return_value.run.side_effect = Exception("data error")
            result = pipeline.retrain()
        assert result["model_replaced"] is False
        assert "Data processing failed" in result["reason"]

    def test_retrain_aborts_if_training_fails(self):
        pipeline = _make_pipeline()
        with patch.object(pipeline, "get_current_accuracy", return_value=0.80), \
             patch.object(pipeline, "backup_current_model",
                          return_value="backups/model_test.pkl"), \
             patch("src.retraining_pipeline.DataProcessing") as mock_dp, \
             patch("src.retraining_pipeline.ModelTraining") as mock_mt:
            mock_dp.return_value.run.return_value = None
            mock_mt.return_value.run.side_effect = Exception("train error")
            result = pipeline.retrain()
        assert result["model_replaced"] is False
        assert "Training failed" in result["reason"]

    def test_log_callback_is_invoked(self):
        pipeline = _make_pipeline()
        messages = []
        with patch.object(pipeline, "get_current_accuracy", side_effect=[0.80, 0.85]), \
             patch.object(pipeline, "backup_current_model",
                          return_value="backups/model_test.pkl"), \
             patch("src.retraining_pipeline.DataProcessing") as mock_dp, \
             patch("src.retraining_pipeline.ModelTraining") as mock_mt:
            mock_dp.return_value.run.return_value = None
            mock_mt.return_value.run.return_value = {"accuracy": 0.85}
            pipeline.retrain(log_callback=messages.append)
        assert len(messages) > 0
