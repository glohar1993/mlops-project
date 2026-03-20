"""
Tier 3 — Batch Predictor
==========================
Offline / scheduled batch scoring pipeline.
- Accepts DataFrame or CSV input
- Processes in configurable chunk sizes
- Writes results with prediction + confidence columns
- Tracks progress via optional callback
"""

import os
import time
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from typing import Optional, Callable, Dict, Any

from src.feature_registry import FEATURE_COLUMNS, LABEL_MAP_REVERSE, encode_operation_mode

BATCH_DIR   = os.getenv("BATCH_RESULTS_DIR", "artifacts/batch_results")
MODEL_PATH  = "artifacts/models/model.pkl"
SCALER_PATH = "artifacts/processed/scaler.pkl"
os.makedirs(BATCH_DIR, exist_ok=True)


class BatchPredictor:
    """Offline batch scoring on DataFrames or CSV files."""

    def __init__(
        self,
        chunk_size:  int = 1000,
        model_path:  str = MODEL_PATH,
        scaler_path: str = SCALER_PATH,
    ):
        self.chunk_size  = chunk_size
        self.model_path  = model_path
        self.scaler_path = scaler_path
        self._model      = None
        self._scaler     = None

    # ── Lazy loading ──────────────────────────────────────────────

    def _load(self) -> None:
        if self._model is None:
            self._model = joblib.load(self.model_path)
        if self._scaler is None:
            self._scaler = joblib.load(self.scaler_path)

    def inject_model(self, model, scaler) -> None:
        """Allow tests to inject mock model/scaler directly."""
        self._model  = model
        self._scaler = scaler

    # ── Core prediction ───────────────────────────────────────────

    def predict_dataframe(
        self,
        df:               pd.DataFrame,
        progress_callback: Optional[Callable[[int, int, int], None]] = None,
    ) -> pd.DataFrame:
        """
        Score an entire DataFrame.

        Adds columns: prediction, prediction_id, confidence
        progress_callback(processed, total, errors)
        """
        self._load()
        chunks  = []
        total   = len(df)
        errors  = 0

        for start in range(0, total, self.chunk_size):
            chunk = df.iloc[start:start + self.chunk_size].copy()
            try:
                # Encode categorical
                if "Operation_Mode" in chunk.columns:
                    chunk["Operation_Mode"] = (
                        chunk["Operation_Mode"].apply(encode_operation_mode)
                    )
                # Inject temporal features if missing
                now = datetime.utcnow()
                for col, default in [
                    ("Year", now.year), ("Month", now.month),
                    ("Day",  now.day),  ("Hour",  now.hour),
                ]:
                    if col not in chunk.columns:
                        chunk[col] = default

                avail = [f for f in FEATURE_COLUMNS if f in chunk.columns]
                X     = chunk[avail].fillna(0).to_numpy(dtype=float)
                Xs    = self._scaler.transform(X)
                preds = self._model.predict(Xs)
                proba = self._model.predict_proba(Xs)

                chunk["prediction"]    = [LABEL_MAP_REVERSE.get(int(p), "Unknown")
                                          for p in preds]
                chunk["prediction_id"] = [int(p) for p in preds]
                chunk["confidence"]    = [round(float(proba[i].max()), 4)
                                          for i in range(len(preds))]
            except Exception as exc:
                chunk["prediction"]    = "ERROR"
                chunk["prediction_id"] = -1
                chunk["confidence"]    = 0.0
                errors += 1

            chunks.append(chunk)
            if progress_callback:
                progress_callback(min(start + self.chunk_size, total), total, errors)

        return pd.concat(chunks, ignore_index=True) if chunks else df.copy()

    # ── CSV convenience ───────────────────────────────────────────

    def predict_csv(
        self,
        input_path:       str,
        output_path:      Optional[str] = None,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Batch predict from CSV → CSV. Returns summary stats."""
        t0 = time.time()
        df = pd.read_csv(input_path)
        result_df = self.predict_dataframe(df, progress_callback)

        if output_path is None:
            ts          = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(BATCH_DIR, f"batch_{ts}.csv")

        result_df.to_csv(output_path, index=False)
        elapsed = time.time() - t0

        dist = {}
        if "prediction" in result_df.columns:
            dist = result_df["prediction"].value_counts().to_dict()

        return {
            "input_rows":               len(df),
            "output_rows":              len(result_df),
            "output_path":              output_path,
            "elapsed_seconds":          round(elapsed, 3),
            "rows_per_second":          round(len(df) / max(0.001, elapsed), 1),
            "prediction_distribution":  dist,
            "errors": int((result_df.get("prediction", pd.Series()) == "ERROR").sum()),
        }
