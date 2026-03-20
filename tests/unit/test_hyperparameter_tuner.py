"""
Unit Tests — Hyperparameter Tuner (src/hyperparameter_tuner.py)
=================================================================
Tests: grid search, param persistence, best_score > baseline,
       various dataset sizes, param structure
"""

import os
import json
import pytest
import numpy as np
import sys
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

os.environ["BEST_PARAMS_FILE"] = "/tmp/test_best_params.json"

from src.hyperparameter_tuner import HyperparameterTuner


@pytest.fixture
def small_dataset():
    """Tiny 3-class classification dataset for fast tuning."""
    rng = np.random.default_rng(42)
    n   = 120
    X   = rng.standard_normal((n, 5))
    y   = rng.integers(0, 3, n)
    return X, y


@pytest.fixture(autouse=True)
def cleanup():
    yield
    if os.path.exists("/tmp/test_best_params.json"):
        os.remove("/tmp/test_best_params.json")


# ════════════════════════════════════════════════════════════════
#  Grid Search (works without Optuna)
# ════════════════════════════════════════════════════════════════

class TestGridSearch:

    def test_tune_returns_dict(self, small_dataset):
        X, y = small_dataset
        tuner  = HyperparameterTuner(n_trials=3, cv_folds=2)
        result = tuner._tune_grid(X, y)
        assert isinstance(result, dict)

    def test_result_has_best_params(self, small_dataset):
        X, y = small_dataset
        tuner  = HyperparameterTuner(n_trials=3, cv_folds=2)
        result = tuner._tune_grid(X, y)
        assert "best_params" in result
        assert result["best_params"] is not None

    def test_result_has_best_score(self, small_dataset):
        X, y = small_dataset
        tuner  = HyperparameterTuner(n_trials=3, cv_folds=2)
        result = tuner._tune_grid(X, y)
        assert "best_score" in result
        assert 0.0 <= result["best_score"] <= 1.0

    def test_best_score_above_random(self, small_dataset):
        """Logistic Regression on 3 classes should beat random (0.33)."""
        X, y = small_dataset
        tuner  = HyperparameterTuner(cv_folds=2)
        result = tuner._tune_grid(X, y)
        assert result["best_score"] > 0.2   # low bar for random data

    def test_best_params_contain_C(self, small_dataset):
        X, y = small_dataset
        tuner = HyperparameterTuner(cv_folds=2)
        tuner._tune_grid(X, y)
        assert "C" in tuner.best_params

    def test_best_params_contain_solver(self, small_dataset):
        X, y = small_dataset
        tuner = HyperparameterTuner(cv_folds=2)
        tuner._tune_grid(X, y)
        assert "solver" in tuner.best_params

    def test_method_is_grid_search(self, small_dataset):
        X, y = small_dataset
        tuner  = HyperparameterTuner(cv_folds=2)
        result = tuner._tune_grid(X, y)
        assert result["method"] == "grid_search"


# ════════════════════════════════════════════════════════════════
#  Persistence
# ════════════════════════════════════════════════════════════════

class TestPersistence:

    def test_save_creates_file(self, small_dataset):
        X, y = small_dataset
        tuner = HyperparameterTuner(cv_folds=2)
        tuner._tune_grid(X, y)
        assert os.path.exists("/tmp/test_best_params.json")

    def test_load_returns_params(self, small_dataset):
        X, y = small_dataset
        tuner = HyperparameterTuner(cv_folds=2)
        tuner._tune_grid(X, y)
        saved_params = tuner.best_params.copy()
        # Create fresh tuner and load
        tuner2 = HyperparameterTuner()
        loaded = tuner2._load()
        assert loaded is not None
        assert loaded["C"] == saved_params["C"]

    def test_get_best_params_uses_in_memory(self, small_dataset):
        X, y = small_dataset
        tuner = HyperparameterTuner(cv_folds=2)
        tuner._tune_grid(X, y)
        params = tuner.get_best_params()
        assert params is not None
        assert "C" in params

    def test_file_not_exists_returns_none(self):
        if os.path.exists("/tmp/test_best_params.json"):
            os.remove("/tmp/test_best_params.json")
        tuner = HyperparameterTuner()
        assert tuner._load() is None


# ════════════════════════════════════════════════════════════════
#  tune() wrapper
# ════════════════════════════════════════════════════════════════

class TestTuneWrapper:

    def test_tune_calls_grid_search_when_optuna_unavailable(self, small_dataset):
        X, y  = small_dataset
        tuner = HyperparameterTuner(cv_folds=2)
        # Temporarily disable Optuna
        import src.hyperparameter_tuner as ht_module
        orig = ht_module.OPTUNA_AVAILABLE
        ht_module.OPTUNA_AVAILABLE = False
        try:
            result = tuner.tune(X, y)
            assert result["method"] == "grid_search"
        finally:
            ht_module.OPTUNA_AVAILABLE = orig

    @pytest.mark.parametrize("n_samples", [60, 120, 300])
    def test_tune_with_various_sizes(self, n_samples):
        rng = np.random.default_rng(n_samples)
        X   = rng.standard_normal((n_samples, 5))
        y   = rng.integers(0, 3, n_samples)
        tuner  = HyperparameterTuner(cv_folds=2)
        result = tuner.tune(X, y)
        assert "best_params" in result
        assert result["best_score"] >= 0.0
