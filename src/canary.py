"""
Tier 1 — Canary Deployment Manager
=====================================
Traffic splitting between stable (A) and canary (B) model versions.

Modes:
  stable     — 100% to stable model
  canary     — X% to canary, (100-X)% to stable
  shadow     — all requests go to stable; canary runs in background (no user impact)
  blue_green — 0% or 100% switch (toggle)

SLO auto-rollback: if canary error rate > 5% OR avg latency > 500ms → auto-revert.
"""

import os
import json
import time
import random
import threading
from datetime import datetime
from typing import Dict, Optional

CANARY_DIR        = "artifacts/canary"
CANARY_STATE_FILE = os.getenv("CANARY_STATE_FILE", os.path.join(CANARY_DIR, "state.json"))
os.makedirs(CANARY_DIR, exist_ok=True)

_lock = threading.Lock()

_DEFAULT_STATE: Dict = {
    "enabled":            False,
    "mode":               "stable",     # stable | canary | shadow | blue_green
    "canary_pct":         0,            # 0-100
    "stable_model_path":  "artifacts/models/model.pkl",
    "canary_model_path":  None,
    "stable_requests":    0,
    "canary_requests":    0,
    "stable_errors":      0,
    "canary_errors":      0,
    "stable_latency_sum": 0.0,
    "canary_latency_sum": 0.0,
    "slo_error_threshold": 0.05,   # 5%
    "slo_latency_ms":      500.0,  # p95 proxy via mean
    "started_at":         None,
    "auto_rollback":      True,
}


class CanaryManager:
    """Manages canary / shadow / blue-green deployment state."""

    def __init__(self, state_file: str = CANARY_STATE_FILE):
        self.state_file = state_file
        os.makedirs(os.path.dirname(state_file), exist_ok=True)
        self._state = self._load()

    # ── Control ──────────────────────────────────────────────────

    def start_canary(self, canary_model_path: str, canary_pct: int = 10,
                     mode: str = "canary") -> Dict:
        """Begin canary deployment."""
        with _lock:
            self._state.update({
                "enabled":            True,
                "mode":               mode,
                "canary_pct":         max(0, min(100, canary_pct)),
                "canary_model_path":  canary_model_path,
                "stable_requests":    0,
                "canary_requests":    0,
                "stable_errors":      0,
                "canary_errors":      0,
                "stable_latency_sum": 0.0,
                "canary_latency_sum": 0.0,
                "started_at":         datetime.utcnow().isoformat(),
            })
            self._save()
        return self._state.copy()

    def stop_canary(self, promote: bool = False) -> Dict:
        """Stop canary. If promote=True, canary becomes the new stable."""
        with _lock:
            if promote and self._state.get("canary_model_path"):
                self._state["stable_model_path"] = self._state["canary_model_path"]
            self._state["enabled"]        = False
            self._state["mode"]           = "stable"
            self._state["canary_pct"]     = 0
            self._save()
        return self._state.copy()

    def increase_traffic(self, new_pct: int) -> Dict:
        """Gradually increase canary traffic percentage."""
        with _lock:
            self._state["canary_pct"] = max(0, min(100, new_pct))
            self._save()
        return self._state.copy()

    # ── Routing ──────────────────────────────────────────────────

    def should_use_canary(self) -> bool:
        """Decide if this request should go to canary model."""
        s = self._state
        if not s.get("enabled"):
            return False
        mode = s.get("mode", "stable")
        if mode == "shadow":
            return False   # Shadow: stable serves; canary runs silently
        if mode in ("canary", "blue_green"):
            return random.randint(1, 100) <= s.get("canary_pct", 0)
        return False

    # ── Metrics ──────────────────────────────────────────────────

    def record_request(self, is_canary: bool, error: bool,
                       latency_ms: float) -> None:
        """Record request outcome; trigger SLO check."""
        with _lock:
            if is_canary:
                self._state["canary_requests"]    += 1
                self._state["canary_latency_sum"] += latency_ms
                if error:
                    self._state["canary_errors"]  += 1
            else:
                self._state["stable_requests"]    += 1
                self._state["stable_latency_sum"] += latency_ms
                if error:
                    self._state["stable_errors"]  += 1
            self._save()

        if self._state.get("auto_rollback"):
            self._check_slo()

    def get_metrics(self) -> Dict:
        s = self._state
        sn = max(1, s["stable_requests"])
        cn = max(1, s["canary_requests"])
        return {
            "enabled":    s["enabled"],
            "mode":       s["mode"],
            "canary_pct": s["canary_pct"],
            "stable": {
                "requests":       s["stable_requests"],
                "error_rate":     round(s["stable_errors"] / sn, 4),
                "avg_latency_ms": round(s["stable_latency_sum"] / sn, 2),
            },
            "canary": {
                "requests":       s["canary_requests"],
                "error_rate":     round(s["canary_errors"] / cn, 4),
                "avg_latency_ms": round(s["canary_latency_sum"] / cn, 2),
            },
        }

    # ── SLO ──────────────────────────────────────────────────────

    def _check_slo(self) -> None:
        s  = self._state
        n  = s["canary_requests"]
        if n < 10:
            return
        err_rate  = s["canary_errors"] / n
        avg_lat   = s["canary_latency_sum"] / n
        if (err_rate > s["slo_error_threshold"] or
                avg_lat > s["slo_latency_ms"]):
            print(f"[Canary] SLO breach — error={err_rate:.2%} "
                  f"latency={avg_lat:.0f}ms → auto-rollback")
            self.stop_canary(promote=False)

    # ── State I/O ────────────────────────────────────────────────

    def _load(self) -> Dict:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file) as f:
                    return {**_DEFAULT_STATE.copy(), **json.load(f)}
            except Exception:
                pass
        return _DEFAULT_STATE.copy()

    def _save(self) -> None:
        with open(self.state_file, "w") as f:
            json.dump(self._state, f, indent=2)

    @property
    def state(self) -> Dict:
        return self._state.copy()
