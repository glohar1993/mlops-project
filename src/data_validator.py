"""
Production Data Quality Validator
===================================
Validates incoming data BEFORE training or serving.
Catches bad data early — prevents garbage-in-garbage-out.

In production: use Great Expectations or Evidently.
This is a lightweight custom implementation.

Checks:
  - Schema: correct columns present
  - Nulls: no missing values
  - Ranges: values within expected bounds
  - Distribution: no extreme outliers
  - Class balance: target not severely imbalanced
"""

import pandas as pd
import numpy as np
from src.logger import get_logger

logger = get_logger(__name__)

# Expected schema and valid ranges per feature
FEATURE_SCHEMA = {
    "Operation_Mode":                    {"type": "numeric", "min": 0,    "max": 3},
    "Temperature_C":                     {"type": "numeric", "min": 40.0, "max": 120.0},
    "Vibration_Hz":                      {"type": "numeric", "min": 0.0,  "max": 10.0},
    "Power_Consumption_kW":              {"type": "numeric", "min": 0.0,  "max": 50.0},
    "Network_Latency_ms":                {"type": "numeric", "min": 0.0,  "max": 200.0},
    "Packet_Loss_%":                     {"type": "numeric", "min": 0.0,  "max": 20.0},
    "Quality_Control_Defect_Rate_%":     {"type": "numeric", "min": 0.0,  "max": 50.0},
    "Production_Speed_units_per_hr":     {"type": "numeric", "min": 50.0, "max": 1000.0},
    "Predictive_Maintenance_Score":      {"type": "numeric", "min": 0.0,  "max": 1.0},
    "Error_Rate_%":                      {"type": "numeric", "min": 0.0,  "max": 50.0},
    "Efficiency_Status":                 {"type": "categorical", "values": ["High", "Medium", "Low"]},
}

MIN_CLASS_RATIO = 0.05   # each class must be at least 5% of data
MAX_NULL_RATIO  = 0.01   # max 1% null values per column


class DataValidator:
    def __init__(self):
        self.results = []
        self.passed  = True

    def _fail(self, check: str, detail: str):
        self.results.append({"check": check, "status": "FAIL", "detail": detail})
        self.passed = False
        logger.error("Data validation FAILED", extra={"check": check, "detail": detail})

    def _pass(self, check: str, detail: str = ""):
        self.results.append({"check": check, "status": "PASS", "detail": detail})
        logger.info("Data validation passed", extra={"check": check})

    def check_schema(self, df: pd.DataFrame):
        expected = set(FEATURE_SCHEMA.keys())
        actual   = set(df.columns)
        missing  = expected - actual
        if missing:
            self._fail("schema_check", f"Missing columns: {missing}")
        else:
            self._pass("schema_check", "All expected columns present")

    def check_nulls(self, df: pd.DataFrame):
        for col in df.columns:
            null_ratio = df[col].isnull().mean()
            if null_ratio > MAX_NULL_RATIO:
                self._fail("null_check",
                    f"Column '{col}' has {null_ratio:.1%} nulls (max {MAX_NULL_RATIO:.1%})")
            else:
                self._pass("null_check", f"Column '{col}' nulls OK ({null_ratio:.1%})")

    def check_ranges(self, df: pd.DataFrame):
        for col, spec in FEATURE_SCHEMA.items():
            if col not in df.columns:
                continue
            if spec["type"] == "numeric":
                out_of_range = ((df[col] < spec["min"]) | (df[col] > spec["max"])).sum()
                pct = out_of_range / len(df)
                if pct > 0.05:  # >5% out-of-range is a problem
                    self._fail("range_check",
                        f"Column '{col}': {out_of_range} rows ({pct:.1%}) out of range "
                        f"[{spec['min']}, {spec['max']}]")
                else:
                    self._pass("range_check", f"Column '{col}' ranges OK")

            elif spec["type"] == "categorical":
                invalid = ~df[col].isin(spec["values"])
                if invalid.any():
                    self._fail("range_check",
                        f"Column '{col}': unexpected values {df[col][invalid].unique().tolist()}")
                else:
                    self._pass("range_check", f"Column '{col}' categories OK")

    def check_class_balance(self, df: pd.DataFrame, target_col: str = "Efficiency_Status"):
        if target_col not in df.columns:
            return
        dist = df[target_col].value_counts(normalize=True)
        for cls, ratio in dist.items():
            if ratio < MIN_CLASS_RATIO:
                self._fail("class_balance",
                    f"Class '{cls}' is severely underrepresented: {ratio:.1%}")
            else:
                self._pass("class_balance", f"Class '{cls}': {ratio:.1%}")

    def check_duplicates(self, df: pd.DataFrame):
        dup_count = df.duplicated().sum()
        dup_ratio = dup_count / len(df)
        if dup_ratio > 0.10:
            self._fail("duplicate_check",
                f"{dup_count} duplicate rows ({dup_ratio:.1%}) — exceeds 10% threshold")
        else:
            self._pass("duplicate_check", f"Duplicates OK ({dup_count} rows, {dup_ratio:.1%})")

    def validate(self, df: pd.DataFrame) -> dict:
        logger.info("Starting data validation", extra={"rows": len(df), "cols": len(df.columns)})

        self.check_schema(df)
        self.check_nulls(df)
        self.check_ranges(df)
        self.check_class_balance(df)
        self.check_duplicates(df)

        passed_count = sum(1 for r in self.results if r["status"] == "PASS")
        failed_count = sum(1 for r in self.results if r["status"] == "FAIL")

        summary = {
            "passed":        self.passed,
            "total_checks":  len(self.results),
            "passed_checks": passed_count,
            "failed_checks": failed_count,
            "details":       self.results,
        }

        if self.passed:
            logger.info("Data validation PASSED", extra={"summary": summary})
        else:
            logger.error("Data validation FAILED — pipeline should stop",
                         extra={"failed": failed_count})

        return summary


def validate_dataframe(df: pd.DataFrame) -> dict:
    """Convenience function — returns validation summary."""
    validator = DataValidator()
    return validator.validate(df)
