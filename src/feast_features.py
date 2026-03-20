"""
Feast Feature Definitions — Finance/Manufacturing Domain
=========================================================
Run this once to register features in the Feast registry:
  feast apply

Then materialize to online store (Redis):
  feast materialize-incremental $(date +%Y-%m-%dT%H:%M:%S)

Feature views defined:
  1. equipment_telemetry_fv   — raw operational features (14)
  2. rolling_stats_fv         — 5m/1h/24h aggregated features
  3. anomaly_score_fv         — isolation forest pre-scores

Entity: equipment_id (str) — anonymized equipment identifier
"""

from datetime import timedelta
from feast import (
    Entity, Feature, FeatureView, FileSource, ValueType, FeatureStore,
    Field, RequestSource
)
from feast.types import Float64, Int64, String

# ── Entity ───────────────────────────────────────────────────────────
equipment = Entity(
    name="equipment_id",
    value_type=ValueType.STRING,
    description="Anonymized equipment identifier (hashed)",
)

# ── Offline Source (S3 Parquet) ───────────────────────────────────────
equipment_telemetry_source = FileSource(
    path="s3://mlops-artifacts-prod-824033490704/feature-store/equipment_telemetry/",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

rolling_stats_source = FileSource(
    path="s3://mlops-artifacts-prod-824033490704/feature-store/rolling_stats/",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

# ── Feature View 1: Raw Operational Telemetry ─────────────────────────
equipment_telemetry_fv = FeatureView(
    name="equipment_telemetry",
    entities=["equipment_id"],
    ttl=timedelta(hours=24),
    schema=[
        Field(name="Operation_Mode",                    dtype=Int64),
        Field(name="Temperature_C",                     dtype=Float64),
        Field(name="Vibration_Hz",                      dtype=Float64),
        Field(name="Power_Consumption_kW",              dtype=Float64),
        Field(name="Network_Latency_ms",                dtype=Float64),
        Field(name="Packet_Loss_pct",                   dtype=Float64),
        Field(name="Quality_Control_Defect_Rate_pct",   dtype=Float64),
        Field(name="Production_Speed_units_per_hr",     dtype=Float64),
        Field(name="Predictive_Maintenance_Score",      dtype=Float64),
        Field(name="Error_Rate_pct",                    dtype=Float64),
    ],
    source=equipment_telemetry_source,
    online=True,
    tags={"domain": "manufacturing", "compliance": "sox", "pii": "false"},
)

# ── Feature View 2: Rolling Statistical Aggregates ────────────────────
rolling_stats_fv = FeatureView(
    name="rolling_stats",
    entities=["equipment_id"],
    ttl=timedelta(hours=6),
    schema=[
        # 5-minute aggregates
        Field(name="temp_mean_5m",      dtype=Float64),
        Field(name="temp_std_5m",       dtype=Float64),
        Field(name="vibration_max_5m",  dtype=Float64),
        Field(name="power_mean_5m",     dtype=Float64),
        Field(name="error_rate_p95_5m", dtype=Float64),
        # 1-hour aggregates
        Field(name="temp_mean_1h",      dtype=Float64),
        Field(name="vibration_trend_1h",dtype=Float64),
        Field(name="power_mean_1h",     dtype=Float64),
        Field(name="defect_rate_1h",    dtype=Float64),
        # 24-hour aggregates
        Field(name="maintenance_score_24h_avg", dtype=Float64),
        Field(name="error_rate_24h_p99",        dtype=Float64),
        Field(name="production_speed_24h_avg",  dtype=Float64),
    ],
    source=rolling_stats_source,
    online=True,
    tags={"domain": "manufacturing", "window": "rolling"},
)

# ── On-Demand Feature View: Temporal Features ─────────────────────────
# These are computed at request time from the raw request fields
# (no offline source needed — injected from current UTC timestamp)
from feast import on_demand_feature_view
import pandas as pd

@on_demand_feature_view(
    sources=[],
    schema=[
        Field(name="hour_sin",  dtype=Float64),
        Field(name="hour_cos",  dtype=Float64),
        Field(name="day_of_week", dtype=Int64),
        Field(name="is_business_hours", dtype=Int64),
    ],
)
def temporal_features(inputs: pd.DataFrame) -> pd.DataFrame:
    """Cyclical encoding of temporal features for model."""
    import numpy as np
    now = pd.Timestamp.utcnow()
    df = pd.DataFrame()
    df["hour_sin"]          = np.sin(2 * np.pi * now.hour / 24)
    df["hour_cos"]          = np.cos(2 * np.pi * now.hour / 24)
    df["day_of_week"]       = now.dayofweek
    df["is_business_hours"] = int(9 <= now.hour <= 17 and now.dayofweek < 5)
    return df
