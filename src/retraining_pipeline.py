"""
Auto Retraining Pipeline
========================
Triggered when drift is detected.
Steps:
  1. Load fresh data
  2. Reprocess it
  3. Retrain model
  4. Compare new vs old model accuracy
  5. If new model is better → replace it (hot-swap)
  6. Reset drift detector window
"""

import joblib
import numpy as np
import os
import shutil
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.data_processing import DataProcessing
from src.model_training import ModelTraining


MODEL_PATH      = "artifacts/models/model.pkl"
SCALER_PATH     = "artifacts/processed/scaler.pkl"
BACKUP_DIR      = "artifacts/models/backups"
RAW_DATA_PATH   = "artifacts/raw/data.csv"
PROCESSED_DIR   = "artifacts/processed"
MODELS_DIR      = "artifacts/models"


class RetrainingPipeline:

    def __init__(self):
        os.makedirs(BACKUP_DIR, exist_ok=True)

    def get_current_accuracy(self) -> float:
        """Evaluate current model on test set."""
        try:
            model  = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            X_test = joblib.load(f"{PROCESSED_DIR}/X_test.pkl")
            y_test = joblib.load(f"{PROCESSED_DIR}/y_test.pkl")
            preds  = model.predict(X_test)
            return round(accuracy_score(y_test, preds), 4)
        except Exception:
            return 0.0

    def backup_current_model(self):
        """Save a timestamped backup of the current model."""
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{BACKUP_DIR}/model_{ts}.pkl"
        if os.path.exists(MODEL_PATH):
            shutil.copy(MODEL_PATH, backup_path)
        return backup_path

    def retrain(self, log_callback=None) -> dict:
        """
        Full retraining cycle. Returns a result dict with:
          - old_accuracy
          - new_accuracy
          - model_replaced (bool)
          - reason
        """
        def log(msg):
            print(f"[Retraining] {msg}")
            if log_callback:
                log_callback(msg)

        result = {
            "timestamp":      datetime.utcnow().isoformat(),
            "old_accuracy":   0.0,
            "new_accuracy":   0.0,
            "model_replaced": False,
            "reason":         "",
        }

        log("Starting retraining pipeline...")

        # Step 1: Get current model accuracy
        old_accuracy = self.get_current_accuracy()
        result["old_accuracy"] = old_accuracy
        log(f"Current model accuracy: {old_accuracy:.2%}")

        # Step 2: Backup current model
        backup = self.backup_current_model()
        log(f"Backed up model to: {backup}")

        # Step 3: Reprocess data
        log("Reprocessing data...")
        try:
            processor = DataProcessing(RAW_DATA_PATH, PROCESSED_DIR)
            processor.run()
        except Exception as e:
            result["reason"] = f"Data processing failed: {e}"
            log(result["reason"])
            return result

        # Step 4: Retrain model
        log("Training new model...")
        try:
            trainer = ModelTraining(PROCESSED_DIR + "/", MODELS_DIR + "/")
            trainer.run()
        except Exception as e:
            result["reason"] = f"Training failed: {e}"
            log(result["reason"])
            return result

        # Step 5: Evaluate new model
        new_accuracy = self.get_current_accuracy()
        result["new_accuracy"] = new_accuracy
        log(f"New model accuracy: {new_accuracy:.2%}")

        # Step 6: Decide whether to keep or rollback
        if new_accuracy >= old_accuracy:
            result["model_replaced"] = True
            result["reason"] = f"New model ({new_accuracy:.2%}) >= old ({old_accuracy:.2%}) — deployed!"
            log(result["reason"])
        else:
            # Rollback to backup
            shutil.copy(backup, MODEL_PATH)
            result["model_replaced"] = False
            result["reason"] = f"New model ({new_accuracy:.2%}) < old ({old_accuracy:.2%}) — rolled back!"
            log(result["reason"])

        return result
