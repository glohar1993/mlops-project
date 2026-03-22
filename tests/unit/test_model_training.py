"""
Unit tests for ModelTraining.
All external I/O (joblib, MLflow, file system) is mocked.
"""
import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, call

# Top-level import for coverage tracking
import src.model_training  # noqa: F401

from src.model_training import ModelTraining


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_trainer(tmp_path):
    """Create a ModelTraining instance with os.makedirs mocked."""
    with patch("os.makedirs"):
        return ModelTraining(
            processed_data_path=str(tmp_path / "processed"),
            model_output_path=str(tmp_path / "models"),
            run_reason="test",
        )


def _inject_data(trainer):
    """Inject synthetic train/test arrays directly into trainer."""
    rng = np.random.default_rng(42)
    trainer.X_train = rng.uniform(0, 1, (80, 14))
    trainer.X_test  = rng.uniform(0, 1, (20, 14))
    trainer.y_train = np.array([i % 3 for i in range(80)])
    trainer.y_test  = np.array([i % 3 for i in range(20)])


# ─────────────────────────────────────────────────────────────────────────────
# TestInit
# ─────────────────────────────────────────────────────────────────────────────

class TestInit:
    def test_init_sets_paths(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        assert trainer.run_reason == "test"
        assert trainer.clf is None

    def test_init_creates_model_dir(self, tmp_path):
        with patch("os.makedirs") as mock_mkdir:
            ModelTraining(
                processed_data_path=str(tmp_path / "processed"),
                model_output_path=str(tmp_path / "models"),
                run_reason="test",
            )
        assert mock_mkdir.called


# ─────────────────────────────────────────────────────────────────────────────
# TestLoadData
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadData:
    def test_load_data_populates_arrays(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        rng = np.random.default_rng(0)
        fake = [
            rng.uniform(0, 1, (80, 14)),  # X_train
            rng.uniform(0, 1, (20, 14)),  # X_test
            np.array([i % 3 for i in range(80)]),  # y_train
            np.array([i % 3 for i in range(20)]),  # y_test
        ]
        with patch("src.model_training.joblib.load", side_effect=fake):
            trainer.load_data()
        assert trainer.X_train.shape == (80, 14)
        assert trainer.X_test.shape  == (20, 14)
        assert len(trainer.y_train) == 80
        assert len(trainer.y_test)  == 20

    def test_load_data_raises_on_file_not_found(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        with patch("src.model_training.joblib.load", side_effect=FileNotFoundError("no file")):
            from src.custom_exception import CustomException
            with pytest.raises(CustomException):
                trainer.load_data()


# ─────────────────────────────────────────────────────────────────────────────
# TestTrainModel
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainModel:
    def test_train_model_returns_params_dict(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        _inject_data(trainer)
        with patch("src.model_training.joblib.dump"):
            params = trainer.train_model()
        assert isinstance(params, dict)
        assert "random_state" in params
        assert "max_iter" in params

    def test_train_model_sets_clf(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        _inject_data(trainer)
        with patch("src.model_training.joblib.dump"):
            trainer.train_model()
        assert trainer.clf is not None

    def test_train_model_clf_is_fitted(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        _inject_data(trainer)
        with patch("src.model_training.joblib.dump"):
            trainer.train_model()
        # A fitted classifier has classes_ attribute
        assert hasattr(trainer.clf, "classes_")

    def test_train_model_saves_pkl(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        _inject_data(trainer)
        with patch("src.model_training.joblib.dump") as mock_dump:
            trainer.train_model()
        mock_dump.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# TestEvaluateModel
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluateModel:
    def _trained_trainer(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        _inject_data(trainer)
        with patch("src.model_training.joblib.dump"):
            trainer.train_model()
        return trainer

    def test_evaluate_returns_dict_with_metrics(self, tmp_path):
        trainer = self._trained_trainer(tmp_path)
        metrics = trainer.evaluate_model()
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics

    def test_accuracy_is_float_in_range(self, tmp_path):
        trainer = self._trained_trainer(tmp_path)
        metrics = trainer.evaluate_model()
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_all_metrics_rounded_to_4(self, tmp_path):
        trainer = self._trained_trainer(tmp_path)
        metrics = trainer.evaluate_model()
        for key in ("accuracy", "precision", "recall", "f1_score"):
            val = metrics[key]
            assert val == round(val, 4)

    def test_model_predicts_three_classes(self, tmp_path):
        trainer = self._trained_trainer(tmp_path)
        preds = trainer.clf.predict(trainer.X_test)
        assert set(preds).issubset({0, 1, 2})


# ─────────────────────────────────────────────────────────────────────────────
# TestRunWithoutMlflow
# ─────────────────────────────────────────────────────────────────────────────

class TestRunWithoutMlflow:
    def test_run_without_mlflow_returns_metrics(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        _inject_data(trainer)
        with patch("src.model_training.MLFLOW_AVAILABLE", False), \
             patch("src.model_training.joblib.dump"):
            result = trainer._run_without_mlflow()
        assert "accuracy" in result

    def test_run_dispatches_to_without_mlflow_when_unavailable(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        _inject_data(trainer)
        with patch("src.model_training.MLFLOW_AVAILABLE", False), \
             patch("src.model_training.joblib.dump"), \
             patch.object(trainer, "load_data"):
            result = trainer.run()
        assert "accuracy" in result


# ─────────────────────────────────────────────────────────────────────────────
# TestGetDvcHash
# ─────────────────────────────────────────────────────────────────────────────

class TestGetDvcHash:
    def test_returns_no_dvc_when_file_missing(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        with patch("os.path.exists", return_value=False):
            result = trainer._get_dvc_hash()
        assert result == "no-dvc"

    def test_returns_hash_when_dvc_file_present(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        dvc_content = {"outs": [{"md5": "abcdef1234567890"}]}
        with patch("os.path.exists", return_value=True), \
             patch("builtins.open", create=True), \
             patch("yaml.safe_load", return_value=dvc_content):
            result = trainer._get_dvc_hash()
        assert result == "abcdef123456"
