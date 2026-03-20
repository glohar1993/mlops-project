"""
Shared test fixtures for all test suites.
"""
import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def sample_raw_df():
    """Minimal valid raw dataframe matching the CSV schema."""
    n = 200
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "Machine_ID":                       [f"M{i:03d}" for i in range(n)],
        "Timestamp":                        pd.date_range("2026-01-01", periods=n, freq="h"),
        "Operation_Mode":                   rng.choice(["Mode_A", "Mode_B", "Mode_C"], n),
        "Temperature_C":                    rng.uniform(60, 100, n),
        "Vibration_Hz":                     rng.uniform(1, 10, n),
        "Power_Consumption_kW":             rng.uniform(30, 90, n),
        "Network_Latency_ms":               rng.uniform(5, 100, n),
        "Packet_Loss_%":                    rng.uniform(0, 6, n),
        "Quality_Control_Defect_Rate_%":    rng.uniform(0, 15, n),
        "Production_Speed_units_per_hr":    rng.uniform(100, 400, n),
        "Predictive_Maintenance_Score":     rng.uniform(0, 1, n),
        "Error_Rate_%":                     rng.uniform(0, 10, n),
        "Efficiency_Status":                rng.choice(["High", "Low", "Medium"], n),
    })


@pytest.fixture
def sample_feature_vector():
    """A single valid prediction request (10 raw features; temporal auto-injected)."""
    return {
        "Operation_Mode":                   1,
        "Temperature_C":                    72.5,
        "Vibration_Hz":                     2.1,
        "Power_Consumption_kW":             45.0,
        "Network_Latency_ms":               12.0,
        "Packet_Loss_%":                    0.5,
        "Quality_Control_Defect_Rate_%":    1.2,
        "Production_Speed_units_per_hr":    320.0,
        "Predictive_Maintenance_Score":     0.85,
        "Error_Rate_%":                     0.8,
        "Year": 2026, "Month": 3, "Day": 19, "Hour": 12,
    }


@pytest.fixture
def anomalous_feature_vector():
    """Feature vector representing anomalous/drift-inducing inputs."""
    return {
        "Operation_Mode":                   2,
        "Temperature_C":                    98.5,
        "Vibration_Hz":                     8.9,
        "Power_Consumption_kW":             89.0,
        "Network_Latency_ms":               95.0,
        "Packet_Loss_%":                    5.5,
        "Quality_Control_Defect_Rate_%":    12.2,
        "Production_Speed_units_per_hr":    120.0,
        "Predictive_Maintenance_Score":     0.15,
        "Error_Rate_%":                     9.8,
        "Year": 2026, "Month": 3, "Day": 19, "Hour": 2,
    }
