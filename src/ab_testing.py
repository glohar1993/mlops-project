"""
Tier 3 — A/B Testing Framework
================================
Statistical comparison of two model versions (A=control, B=treatment).
- Chi-square test on prediction distributions
- Error rate and latency comparison
- Winner selection based on statistical significance (p < 0.05)
"""

import os
import json
import math
import random
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple

AB_DIR        = "artifacts/ab_test"
AB_STATE_FILE = os.getenv("AB_STATE_FILE", os.path.join(AB_DIR, "state.json"))
os.makedirs(AB_DIR, exist_ok=True)

_lock = threading.Lock()


# ── Chi-square helpers ────────────────────────────────────────────────────────

def _chi2_sf(chi2: float, df: int) -> float:
    """Survival P(X > chi2) for chi-square distribution."""
    try:
        import scipy.stats
        return float(scipy.stats.chi2.sf(chi2, df))
    except ImportError:
        # Approximation: df=2 exact; others rough
        if df == 2:
            return math.exp(-chi2 / 2)
        return max(0.0, min(1.0, math.exp(-chi2 / (2 * df))))


def chi_square_test(
    obs_a: List[int],
    obs_b: List[int],
) -> Tuple[float, float]:
    """Chi-square goodness-of-fit test between two count distributions."""
    n_a = sum(obs_a)
    n_b = sum(obs_b)
    if n_a == 0 or n_b == 0:
        return 0.0, 1.0
    chi2 = 0.0
    for a, b in zip(obs_a, obs_b):
        total    = a + b
        exp_a    = total * n_a / (n_a + n_b)
        if exp_a > 0:
            chi2 += (a - exp_a) ** 2 / exp_a
    df      = max(1, len(obs_a) - 1)
    p_value = _chi2_sf(chi2, df)
    return round(chi2, 4), round(p_value, 4)


# ── A/B Test ──────────────────────────────────────────────────────────────────

class ABTest:
    """Track and compare two model variants over live traffic."""

    MIN_SAMPLES_FOR_ANALYSIS = 30

    def __init__(
        self,
        test_name:       str,
        model_a_path:    str,
        model_b_path:    str,
        traffic_split_b: int = 50,   # % traffic to B
    ):
        self.test_name       = test_name
        self.model_a_path    = model_a_path
        self.model_b_path    = model_b_path
        self.traffic_split_b = max(0, min(100, traffic_split_b))

        self._state: Dict = {
            "test_name":        test_name,
            "started_at":       datetime.utcnow().isoformat(),
            "traffic_split_b":  self.traffic_split_b,
            "model_a": {"preds": [0, 0, 0], "latencies": [], "errors": 0, "n": 0},
            "model_b": {"preds": [0, 0, 0], "latencies": [], "errors": 0, "n": 0},
            "status":  "running",
            "winner":  None,
        }

    def assign(self) -> str:
        """Route this request to 'a' or 'b'."""
        return "b" if random.randint(1, 100) <= self.traffic_split_b else "a"

    def record(self, variant: str, prediction: int,
               latency_ms: float, error: bool = False) -> None:
        """Record outcome for variant 'a' or 'b'."""
        with _lock:
            m = self._state[f"model_{variant}"]
            m["n"] += 1
            if 0 <= prediction <= 2:
                m["preds"][prediction] += 1
            m["latencies"].append(round(latency_ms, 2))
            if error:
                m["errors"] += 1

    def analyze(self) -> Dict:
        """Run statistical analysis. Returns winner if enough data."""
        a = self._state["model_a"]
        b = self._state["model_b"]
        n_a, n_b = a["n"], b["n"]

        if n_a < self.MIN_SAMPLES_FOR_ANALYSIS or n_b < self.MIN_SAMPLES_FOR_ANALYSIS:
            return {
                "status":       "insufficient_data",
                "n_a":          n_a,
                "n_b":          n_b,
                "min_required": self.MIN_SAMPLES_FOR_ANALYSIS,
            }

        chi2, p_value = chi_square_test(a["preds"], b["preds"])

        avg_a = sum(a["latencies"]) / len(a["latencies"])
        avg_b = sum(b["latencies"]) / len(b["latencies"])
        err_a = a["errors"] / n_a
        err_b = b["errors"] / n_b

        significant = p_value < 0.05
        winner: Optional[str] = None
        if significant:
            winner = "b" if err_b <= err_a else "a"

        self._state["status"] = "complete" if significant else "running"
        self._state["winner"] = winner

        return {
            "status":                   self._state["status"],
            "n_a":                      n_a,
            "n_b":                      n_b,
            "chi_square":               chi2,
            "p_value":                  p_value,
            "statistically_significant": significant,
            "model_a": {
                "error_rate":     round(err_a, 4),
                "avg_latency_ms": round(avg_a, 2),
                "pred_dist":      {"High": a["preds"][0], "Low": a["preds"][1],
                                   "Medium": a["preds"][2]},
            },
            "model_b": {
                "error_rate":     round(err_b, 4),
                "avg_latency_ms": round(avg_b, 2),
                "pred_dist":      {"High": b["preds"][0], "Low": b["preds"][1],
                                   "Medium": b["preds"][2]},
            },
            "winner":         winner,
            "recommendation": (
                f"Promote model_{winner} to stable"
                if winner else "Continue collecting data"
            ),
        }

    def to_dict(self) -> Dict:
        return self._state.copy()
