"""
Feature Store Integration
=========================
Finance domain MLOps — feature retrieval layer.

Architecture:
  Online (Redis)   → sub-ms retrieval for real-time inference
  Offline (S3)     → batch retrieval for model training

Feature Groups:
  equipment_features    — operational telemetry (14 features)
  rolling_aggregates    — 5m/1h/24h statistical windows
  anomaly_scores        — pre-computed isolation forest scores

Usage in application.py:
  from src.feature_store import FeatureStoreClient
  fs = FeatureStoreClient()
  features = fs.get_online_features(entity_id="equipment-001")
"""

import os
import json
import redis
import hashlib
import numpy as np
from datetime import datetime
from typing import Optional

REDIS_URL      = os.getenv("REDIS_URL", "redis://:feast-redis-secret@redis-feature-store:6379")
FEATURE_TTL    = int(os.getenv("FEATURE_TTL_SECONDS", "300"))   # 5 minutes
FEATURE_PREFIX = "mlops:features:"

# The 14 features the model expects
FEATURE_NAMES = [
    "Operation_Mode", "Temperature_C", "Vibration_Hz",
    "Power_Consumption_kW", "Network_Latency_ms", "Packet_Loss_%",
    "Quality_Control_Defect_Rate_%", "Production_Speed_units_per_hr",
    "Predictive_Maintenance_Score", "Error_Rate_%",
    "Year", "Month", "Day", "Hour"
]


class FeatureStoreClient:
    """
    Thin wrapper around Redis for online feature serving.
    Falls back to direct computation if Redis is unavailable.
    """

    def __init__(self):
        self._redis: Optional[redis.Redis] = None
        self._connected = False
        self._connect()

    def _connect(self):
        try:
            self._redis = redis.from_url(REDIS_URL, socket_timeout=0.05, decode_responses=True)
            self._redis.ping()
            self._connected = True
            print("[FeatureStore] Connected to Redis online store")
        except Exception as e:
            print(f"[FeatureStore] Redis unavailable: {e} — using passthrough mode")
            self._connected = False

    # ------------------------------------------------------------------ #
    #  Write features to online store
    # ------------------------------------------------------------------ #
    def put_features(self, entity_id: str, features: dict) -> bool:
        """Write feature vector to Redis with TTL."""
        if not self._connected:
            return False
        try:
            key = f"{FEATURE_PREFIX}{entity_id}"
            payload = json.dumps({
                **features,
                "_ts":      datetime.utcnow().isoformat(),
                "_entity":  entity_id,
            })
            self._redis.setex(key, FEATURE_TTL, payload)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    #  Read features from online store
    # ------------------------------------------------------------------ #
    def get_online_features(self, entity_id: str) -> Optional[dict]:
        """
        Retrieve feature vector from Redis.
        Returns None if entity not found or Redis is down.
        """
        if not self._connected:
            return None
        try:
            key = f"{FEATURE_PREFIX}{entity_id}"
            raw = self._redis.get(key)
            return json.loads(raw) if raw else None
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    #  Build feature vector from raw request + store in online store
    # ------------------------------------------------------------------ #
    def build_and_store(self, entity_id: str, raw_data: dict) -> dict:
        """
        Auto-inject temporal features and store in online store.
        Returns the enriched feature dict.
        """
        now = datetime.utcnow()
        enriched = {
            **raw_data,
            "Year":  raw_data.get("Year",  now.year),
            "Month": raw_data.get("Month", now.month),
            "Day":   raw_data.get("Day",   now.day),
            "Hour":  raw_data.get("Hour",  now.hour),
        }
        self.put_features(entity_id, enriched)
        return enriched

    # ------------------------------------------------------------------ #
    #  Get ordered feature vector for model inference
    # ------------------------------------------------------------------ #
    def extract_feature_vector(self, features: dict) -> list:
        """Return list of float values in model-expected order."""
        return [float(features.get(f, 0.0)) for f in FEATURE_NAMES]

    # ------------------------------------------------------------------ #
    #  Compute entity ID from request (hashed for anonymization)
    # ------------------------------------------------------------------ #
    @staticmethod
    def entity_id_from_request(data: dict) -> str:
        """
        For finance/compliance: entity ID is a deterministic hash of
        stable fields (not PII). Equipment identified by operation mode
        + hash of first 4 numeric features.
        """
        key_fields = str({
            "mode": data.get("Operation_Mode", 0),
            "temp": round(float(data.get("Temperature_C", 0)), 0),
        })
        return hashlib.sha256(key_fields.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------ #
    #  Feature statistics for drift monitoring (per entity)
    # ------------------------------------------------------------------ #
    def get_feature_stats(self, entity_ids: list) -> dict:
        """
        Batch fetch features and compute aggregate stats.
        Used by drift detector to compare online vs offline distributions.
        """
        if not self._connected or not entity_ids:
            return {}

        pipeline = self._redis.pipeline()
        for eid in entity_ids:
            pipeline.get(f"{FEATURE_PREFIX}{eid}")
        results = pipeline.execute()

        vectors = []
        for raw in results:
            if raw:
                try:
                    d = json.loads(raw)
                    vec = [float(d.get(f, 0.0)) for f in FEATURE_NAMES]
                    vectors.append(vec)
                except Exception:
                    pass

        if not vectors:
            return {}

        arr = np.array(vectors)
        return {
            "count":    len(vectors),
            "mean":     arr.mean(axis=0).tolist(),
            "std":      arr.std(axis=0).tolist(),
            "min":      arr.min(axis=0).tolist(),
            "max":      arr.max(axis=0).tolist(),
            "features": FEATURE_NAMES,
            "ts":       datetime.utcnow().isoformat(),
        }

    def health(self) -> dict:
        try:
            if self._connected:
                self._redis.ping()
                return {"status": "healthy", "backend": "redis"}
        except Exception:
            self._connected = False
        return {"status": "degraded", "backend": "passthrough"}
