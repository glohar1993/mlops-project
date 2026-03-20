"""
Production-Grade Model Training with MLflow Experiment Tracking
===============================================================
Every training run is tracked in MLflow:
  - Parameters (model hyperparams)
  - Metrics (accuracy, precision, recall, F1)
  - Artifacts (model.pkl, scaler.pkl)
  - Tags (run reason, git commit, data version)

In production: MLflow server stores runs in S3 + PostgreSQL.
Locally: runs stored in ./mlruns directory.
"""

import os
import joblib
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

# MLflow is optional — gracefully skip if not installed
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    try:
        mlflow.set_experiment("mlops-efficiency-prediction")
    except Exception as exp_err:
        logger.warning(f"MLflow experiment setup failed (continuing): {exp_err}")
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.info("MLflow not installed — skipping experiment tracking")


class ModelTraining:
    def __init__(self, processed_data_path, model_output_path,
                 run_reason: str = "scheduled"):
        self.processed_path = processed_data_path
        self.model_path     = model_output_path
        self.run_reason     = run_reason
        self.clf            = None
        self.X_train = self.X_test = self.y_train = self.y_test = None

        os.makedirs(self.model_path, exist_ok=True)
        logger.info("ModelTraining initialised", extra={"run_reason": run_reason})

    def load_data(self):
        try:
            self.X_train = joblib.load(os.path.join(self.processed_path, "X_train.pkl"))
            self.X_test  = joblib.load(os.path.join(self.processed_path, "X_test.pkl"))
            self.y_train = joblib.load(os.path.join(self.processed_path, "y_train.pkl"))
            self.y_test  = joblib.load(os.path.join(self.processed_path, "y_test.pkl"))
            logger.info("Training data loaded", extra={
                "train_size": len(self.X_train),
                "test_size":  len(self.X_test)
            })
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise CustomException("Failed to load data", e)

    def train_model(self):
        try:
            params = {"random_state": 42, "max_iter": 1000, "C": 1.0}
            self.clf = LogisticRegression(**params)
            self.clf.fit(self.X_train, self.y_train)
            joblib.dump(self.clf, os.path.join(self.model_path, "model.pkl"))
            logger.info("Model trained and saved", extra={"params": params})
            return params
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise CustomException("Failed to train model", e)

    def evaluate_model(self) -> dict:
        try:
            y_pred = self.clf.predict(self.X_test)
            metrics = {
                "accuracy":  round(accuracy_score(self.y_test, y_pred),  4),
                "precision": round(precision_score(self.y_test, y_pred, average="weighted"), 4),
                "recall":    round(recall_score(self.y_test, y_pred,    average="weighted"), 4),
                "f1_score":  round(f1_score(self.y_test, y_pred,        average="weighted"), 4),
            }
            logger.info("Model evaluation complete", extra={"metrics": metrics})
            return metrics
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise CustomException("Failed to evaluate model", e)

    def run(self) -> dict:
        self.load_data()

        if MLFLOW_AVAILABLE:
            return self._run_with_mlflow()
        else:
            return self._run_without_mlflow()

    def _run_with_mlflow(self) -> dict:
        with mlflow.start_run(run_name=f"train_{self.run_reason}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}") as run:
            # Tags
            mlflow.set_tags({
                "run_reason":    self.run_reason,
                "model_type":    "LogisticRegression",
                "data_version":  self._get_dvc_hash(),
                "environment":   os.getenv("ENVIRONMENT", "local"),
                "git_commit":    os.getenv("GIT_COMMIT", "unknown"),
            })

            # Train (saves model.pkl locally first — S3 upload is separate below)
            params = self.train_model()
            mlflow.log_params(params)

            # Evaluate
            metrics = self.evaluate_model()
            mlflow.log_metrics(metrics)

            # Upload artifacts to S3 via MLflow — gracefully degrade if S3 is unreachable.
            # The local model.pkl is already written by train_model(); retraining can
            # succeed (hot-swap from local file) even if this upload fails.
            try:
                mlflow.sklearn.log_model(self.clf, "model",
                    registered_model_name="mlops-efficiency-predictor")
                mlflow.log_artifact(os.path.join(self.processed_path, "scaler.pkl"))
            except Exception as s3_err:
                logger.warning(
                    f"S3/MLflow artifact upload failed — model saved locally only. "
                    f"Reason: {s3_err}"
                )

            run_id = run.info.run_id
            logger.info("MLflow run complete", extra={
                "run_id": run_id, "metrics": metrics
            })

            metrics["mlflow_run_id"] = run_id
            return metrics

    def _get_dvc_hash(self) -> str:
        """Return DVC content hash of training data for reproducibility tracking."""
        dvc_file = "artifacts/raw/data.csv.dvc"
        if os.path.exists(dvc_file):
            try:
                import yaml
                with open(dvc_file) as f:
                    meta = yaml.safe_load(f)
                return meta.get("outs", [{}])[0].get("md5", "no-dvc")[:12]
            except Exception:
                pass
        return "no-dvc"

    def _run_without_mlflow(self) -> dict:
        self.train_model()
        return self.evaluate_model()


if __name__ == "__main__":
    trainer = ModelTraining("artifacts/processed/", "artifacts/models/",
                            run_reason="manual")
    result = trainer.run()
    print(result)
