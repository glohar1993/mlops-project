"""
Feature Registry — Single Source of Truth
==========================================
ALL feature definitions, label mappings, and transformations
live here. Every other module imports from this file.

Previously features were defined in 4 separate places:
  - data_processing.py   (lines 58-62)
  - dag_1_daily_training.py  (lines 115-121)
  - dag_2_drift_retraining.py (lines 156-162)
  - application.py  (lines 78-84)

This eliminates training-serving skew — the single most dangerous
failure mode in production ML systems.
"""

# ── Feature columns (model-expected order) ───────────────────────────
FEATURE_COLUMNS = [
    "Operation_Mode",
    "Temperature_C",
    "Vibration_Hz",
    "Power_Consumption_kW",
    "Network_Latency_ms",
    "Packet_Loss_%",
    "Quality_Control_Defect_Rate_%",
    "Production_Speed_units_per_hr",
    "Predictive_Maintenance_Score",
    "Error_Rate_%",
    "Year",
    "Month",
    "Day",
    "Hour",
]

# ── Target / labels ───────────────────────────────────────────────────
TARGET_COLUMN = "Efficiency_Status"

# Fixed mapping — never use LabelEncoder.fit_transform() in production.
# LabelEncoder is non-deterministic: if the training data changes and
# one class disappears, label 0/1/2 assignments shift silently.
LABEL_MAP = {"High": 0, "Low": 1, "Medium": 2}
LABEL_MAP_REVERSE = {v: k for k, v in LABEL_MAP.items()}

# ── Categorical columns with fixed encodings ──────────────────────────
# Operation_Mode: defined values → numeric codes (extend as needed)
OPERATION_MODE_MAP = {
    "Mode_A": 0,
    "Mode_B": 1,
    "Mode_C": 2,
    "0":      0,   # already-encoded passthrough
    "1":      1,
    "2":      2,
}

# ── Data quality thresholds ───────────────────────────────────────────
MIN_ROWS              = 100
MAX_NULL_PCT          = 10.0    # % per column
REQUIRED_RAW_COLUMNS  = [
    "Operation_Mode", "Temperature_C", "Vibration_Hz",
    "Power_Consumption_kW", "Network_Latency_ms", "Packet_Loss_%",
    "Quality_Control_Defect_Rate_%", "Production_Speed_units_per_hr",
    "Predictive_Maintenance_Score", "Error_Rate_%",
    "Efficiency_Status", "Timestamp",
]

# ── Model quality thresholds ──────────────────────────────────────────
MIN_ACCURACY     = 0.75    # Below this → do not promote to Production
MIN_ACCURACY_GAIN = 0.02   # Drift-retrain: new model must beat baseline by 2%

# ── Drift thresholds (PSI) ────────────────────────────────────────────
PSI_WARNING  = 0.1
PSI_CRITICAL = 0.2
DRIFT_WINDOW = 50          # Rolling window size for drift detection


# ── Helper: encode Operation_Mode safely ─────────────────────────────
def encode_operation_mode(value) -> int:
    """
    Convert Operation_Mode to integer using fixed mapping.
    Falls back to 0 for unknown values instead of raising.
    """
    str_val = str(value).strip()
    if str_val in OPERATION_MODE_MAP:
        return OPERATION_MODE_MAP[str_val]
    # Numeric passthrough (already an int)
    try:
        return int(float(str_val))
    except (ValueError, TypeError):
        return 0


def apply_label_map(series):
    """Map target column using fixed LABEL_MAP. Raises on unknown values."""
    import pandas as pd
    mapped = series.map(LABEL_MAP)
    unmapped = series[mapped.isna()].unique()
    if len(unmapped) > 0:
        raise ValueError(
            f"Unknown {TARGET_COLUMN} values: {unmapped.tolist()}. "
            f"Expected one of: {list(LABEL_MAP.keys())}"
        )
    return mapped.astype(int)
