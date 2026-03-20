"""
Tier 3 — Hyperparameter Tuner (Optuna + Grid Search fallback)
==============================================================
Automated search for best LogisticRegression hyperparameters.
- Uses Optuna (Bayesian optimisation) when installed
- Falls back to grid search otherwise
- Persists best params to artifacts/models/best_params.json
"""

import os
import json
import numpy as np
from typing import Dict, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

BEST_PARAMS_FILE = os.getenv("BEST_PARAMS_FILE", "artifacts/models/best_params.json")
os.makedirs(os.path.dirname(BEST_PARAMS_FILE), exist_ok=True)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class HyperparameterTuner:
    """Tune LogisticRegression using Optuna or grid search."""

    # Fallback grid
    _GRID = {
        "C":       [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "solver":  ["lbfgs", "saga"],
        "max_iter": [500, 1000],
    }

    def __init__(self, n_trials: int = 20, cv_folds: int = 3,
                 random_state: int = 42):
        self.n_trials     = n_trials
        self.cv_folds     = cv_folds
        self.random_state = random_state
        self.best_params:  Optional[Dict[str, Any]] = None
        self.best_score:   float = 0.0
        self.method_used:  str   = "none"

    def tune(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Run hyperparameter search. Returns result dict."""
        if OPTUNA_AVAILABLE:
            return self._tune_optuna(X_train, y_train)
        return self._tune_grid(X_train, y_train)

    def _tune_optuna(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        def objective(trial: "optuna.Trial") -> float:
            params = {
                "C":            trial.suggest_float("C", 1e-3, 1e2, log=True),
                "max_iter":     trial.suggest_int("max_iter", 200, 1000, step=100),
                "solver":       trial.suggest_categorical("solver", ["lbfgs", "saga"]),
                "penalty":      "l2",
                "random_state": self.random_state,
            }
            clf    = LogisticRegression(**params)
            scores = cross_val_score(clf, X, y, cv=self.cv_folds,
                                     scoring="accuracy", n_jobs=1)
            return float(scores.mean())

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        self.best_params = {**study.best_params, "random_state": self.random_state,
                            "penalty": "l2"}
        self.best_score  = study.best_value
        self.method_used = "optuna"
        self._save()
        return {
            "best_params": self.best_params,
            "best_score":  round(self.best_score, 4),
            "method":      "optuna",
            "n_trials":    self.n_trials,
        }

    def _tune_grid(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        best_score  = 0.0
        best_params = {}
        for C in self._GRID["C"]:
            for solver in self._GRID["solver"]:
                for max_iter in self._GRID["max_iter"]:
                    try:
                        clf    = LogisticRegression(C=C, solver=solver,
                                                    max_iter=max_iter,
                                                    random_state=self.random_state)
                        scores = cross_val_score(clf, X, y, cv=self.cv_folds,
                                                 scoring="accuracy")
                        score  = float(scores.mean())
                        if score > best_score:
                            best_score  = score
                            best_params = {"C": C, "solver": solver,
                                           "max_iter": max_iter,
                                           "random_state": self.random_state}
                    except Exception:
                        continue

        self.best_params = best_params
        self.best_score  = best_score
        self.method_used = "grid_search"
        self._save()
        return {
            "best_params": best_params,
            "best_score":  round(best_score, 4),
            "method":      "grid_search",
        }

    def get_best_params(self) -> Optional[Dict[str, Any]]:
        if self.best_params:
            return self.best_params
        return self._load()

    def _save(self) -> None:
        with open(BEST_PARAMS_FILE, "w") as f:
            json.dump({"best_params": self.best_params,
                       "best_score":  self.best_score,
                       "method":      self.method_used}, f, indent=2)

    def _load(self) -> Optional[Dict[str, Any]]:
        if os.path.exists(BEST_PARAMS_FILE):
            try:
                with open(BEST_PARAMS_FILE) as f:
                    return json.load(f).get("best_params")
            except Exception:
                pass
        return None
