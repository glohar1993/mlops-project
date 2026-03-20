"""
Unit Tests — Canary Deployment Manager (src/canary.py)
========================================================
Tests: start/stop, traffic split, SLO auto-rollback, promote,
       shadow mode, blue-green, metrics tracking
"""

import os
import pytest
import tempfile
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.canary import CanaryManager


@pytest.fixture
def canary(tmp_path):
    """Fresh canary manager with isolated state file."""
    state_file = str(tmp_path / "canary_state.json")
    return CanaryManager(state_file=state_file)


# ════════════════════════════════════════════════════════════════
#  Start / Stop
# ════════════════════════════════════════════════════════════════

class TestCanaryStartStop:

    def test_default_state_is_stable(self, canary):
        assert canary.state["enabled"] is False
        assert canary.state["mode"] == "stable"

    def test_start_canary_enables(self, canary):
        canary.start_canary("models/new.pkl", canary_pct=10)
        assert canary.state["enabled"] is True
        assert canary.state["mode"] == "canary"
        assert canary.state["canary_pct"] == 10

    def test_start_sets_model_path(self, canary):
        canary.start_canary("models/v2.pkl", canary_pct=20)
        assert canary.state["canary_model_path"] == "models/v2.pkl"

    def test_stop_disables_canary(self, canary):
        canary.start_canary("models/v2.pkl", canary_pct=30)
        canary.stop_canary()
        assert canary.state["enabled"] is False
        assert canary.state["mode"] == "stable"

    def test_stop_promote_updates_stable_path(self, canary):
        canary.start_canary("models/v2.pkl", canary_pct=50)
        canary.stop_canary(promote=True)
        assert canary.state["stable_model_path"] == "models/v2.pkl"

    def test_stop_no_promote_keeps_stable_path(self, canary):
        original_stable = canary.state["stable_model_path"]
        canary.start_canary("models/v2.pkl", canary_pct=50)
        canary.stop_canary(promote=False)
        assert canary.state["stable_model_path"] == original_stable

    def test_increase_traffic(self, canary):
        canary.start_canary("models/v2.pkl", canary_pct=10)
        canary.increase_traffic(50)
        assert canary.state["canary_pct"] == 50

    def test_traffic_clamped_to_100(self, canary):
        canary.start_canary("models/v2.pkl", canary_pct=10)
        canary.increase_traffic(200)
        assert canary.state["canary_pct"] == 100

    def test_traffic_clamped_to_zero(self, canary):
        canary.start_canary("models/v2.pkl", canary_pct=10)
        canary.increase_traffic(-50)
        assert canary.state["canary_pct"] == 0


# ════════════════════════════════════════════════════════════════
#  Routing
# ════════════════════════════════════════════════════════════════

class TestCanaryRouting:

    def test_no_canary_always_returns_false(self, canary):
        for _ in range(20):
            assert canary.should_use_canary() is False

    def test_shadow_mode_never_routes_to_canary(self, canary):
        canary.start_canary("models/v2.pkl", canary_pct=100, mode="shadow")
        for _ in range(20):
            assert canary.should_use_canary() is False

    def test_100pct_canary_always_routes_to_canary(self, canary):
        canary.start_canary("models/v2.pkl", canary_pct=100, mode="canary")
        for _ in range(20):
            assert canary.should_use_canary() is True

    def test_0pct_canary_never_routes_to_canary(self, canary):
        canary.start_canary("models/v2.pkl", canary_pct=0, mode="canary")
        for _ in range(20):
            assert canary.should_use_canary() is False

    def test_50pct_split_is_roughly_half(self, canary):
        """Statistical test: 50% split ≈ 50% canary over 1000 trials."""
        canary.start_canary("models/v2.pkl", canary_pct=50, mode="canary")
        canary_count = sum(canary.should_use_canary() for _ in range(1000))
        # Allow ±10% variance
        assert 350 <= canary_count <= 650


# ════════════════════════════════════════════════════════════════
#  SLO Auto-Rollback
# ════════════════════════════════════════════════════════════════

class TestCanarySLO:

    def test_slo_breach_triggers_rollback(self, canary):
        """Error rate > 5% after 10 canary requests → auto-rollback."""
        canary.start_canary("models/v2.pkl", canary_pct=100, mode="canary")
        # Inject 10 canary errors
        for _ in range(10):
            canary.record_request(is_canary=True, error=True, latency_ms=10.0)
        assert canary.state["enabled"] is False   # rolled back

    def test_latency_breach_triggers_rollback(self, canary):
        """Average latency > 500ms → auto-rollback."""
        canary.start_canary("models/v2.pkl", canary_pct=100, mode="canary")
        for _ in range(10):
            canary.record_request(is_canary=True, error=False, latency_ms=600.0)
        assert canary.state["enabled"] is False

    def test_healthy_canary_not_rolled_back(self, canary):
        canary.start_canary("models/v2.pkl", canary_pct=50, mode="canary")
        for _ in range(20):
            canary.record_request(is_canary=True, error=False, latency_ms=50.0)
        assert canary.state["enabled"] is True   # still running

    def test_stable_requests_not_affect_slo(self, canary):
        canary.start_canary("models/v2.pkl", canary_pct=50, mode="canary")
        for _ in range(10):
            canary.record_request(is_canary=False, error=True, latency_ms=600.0)
        assert canary.state["enabled"] is True   # stable errors don't rollback canary


# ════════════════════════════════════════════════════════════════
#  Metrics
# ════════════════════════════════════════════════════════════════

class TestCanaryMetrics:

    def test_metrics_track_requests(self, canary):
        canary.start_canary("models/v2.pkl", canary_pct=50, mode="canary")
        canary.record_request(is_canary=True,  error=False, latency_ms=30.0)
        canary.record_request(is_canary=False, error=False, latency_ms=25.0)
        m = canary.get_metrics()
        assert m["canary"]["requests"] == 1
        assert m["stable"]["requests"] == 1

    def test_error_rate_calculation(self, canary):
        canary.start_canary("models/v2.pkl", canary_pct=50, mode="canary")
        canary.record_request(is_canary=True, error=True,  latency_ms=10.0)
        canary.record_request(is_canary=True, error=False, latency_ms=10.0)
        m = canary.get_metrics()
        assert m["canary"]["error_rate"] == pytest.approx(0.5)

    def test_avg_latency_calculation(self, canary):
        canary.start_canary("models/v2.pkl", canary_pct=50, mode="canary")
        canary.record_request(is_canary=True, error=False, latency_ms=100.0)
        canary.record_request(is_canary=True, error=False, latency_ms=200.0)
        m = canary.get_metrics()
        assert m["canary"]["avg_latency_ms"] == pytest.approx(150.0)
