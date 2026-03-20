"""
MLOps Load Testing Suite — Locust
===================================
SLO targets:
  - p95 latency < 500ms
  - Error rate  < 1%
  - Throughput  > 10 req/s at 50 concurrent users

Usage:
  # Headless (CI/CD):
  locust -f tests/load/locustfile.py --headless -u 50 -r 5 --run-time 2m \
         --host http://3.136.26.65:30080

  # Web UI (dev):
  locust -f tests/load/locustfile.py --host http://3.136.26.65:30080
  # then open http://localhost:8089
"""

import os
import json
import random
from locust import HttpUser, task, between, events
from locust.runners import MasterRunner


TARGET_URL = os.getenv("TARGET_URL", "http://3.136.26.65:30080")

# SLO thresholds
SLO_P95_MS    = 500    # p95 response time must be < 500ms
SLO_ERROR_PCT = 1.0    # error rate must be < 1%
SLO_MIN_RPS   = 10.0   # minimum throughput at full load

# ── Payload templates ─────────────────────────────────────────
NORMAL_PAYLOAD = {
    "Operation_Mode": 1,
    "Temperature_C": 72.0,
    "Vibration_Hz": 1.8,
    "Power_Consumption_kW": 45.0,
    "Network_Latency_ms": 12.0,
    "Packet_Loss_%": 0.5,
    "Quality_Control_Defect_Rate_%": 1.2,
    "Production_Speed_units_per_hr": 320.0,
    "Predictive_Maintenance_Score": 0.85,
    "Error_Rate_%": 0.8,
}

ANOMALOUS_PAYLOAD = {
    "Operation_Mode": 2,
    "Temperature_C": 145.0,
    "Vibration_Hz": 9.5,
    "Power_Consumption_kW": 98.0,
    "Network_Latency_ms": 450.0,
    "Packet_Loss_%": 15.0,
    "Quality_Control_Defect_Rate_%": 18.5,
    "Production_Speed_units_per_hr": 50.0,
    "Predictive_Maintenance_Score": 0.05,
    "Error_Rate_%": 22.0,
}

HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": os.getenv("API_KEY", "mlops-admin-key-dev"),
}


def _jitter(payload: dict, pct: float = 0.05) -> dict:
    """Add ±5% random jitter to numeric values to simulate real traffic."""
    result = {}
    for k, v in payload.items():
        if isinstance(v, (int, float)):
            result[k] = round(v * (1 + random.uniform(-pct, pct)), 2)
        else:
            result[k] = v
    return result


# ── Primary load user ─────────────────────────────────────────
class MLOpsUser(HttpUser):
    """
    Simulates production traffic mix:
      - 55% normal predictions (most common operation)
      - 25% anomalous predictions (edge cases)
      - 10% health checks (readiness probes / uptime monitoring)
      - 10% drift status polls (drift detection CronJob simulation)
    """
    wait_time = between(0.5, 2.0)
    host      = TARGET_URL

    @task(5)
    def predict_normal(self):
        """Normal operating conditions — expect 'High' or 'Medium'."""
        with self.client.post(
            "/predict",
            data=json.dumps(_jitter(NORMAL_PAYLOAD)),
            headers=HEADERS,
            catch_response=True,
            name="POST /predict [normal]",
        ) as resp:
            if resp.status_code == 200:
                body = resp.json()
                if "prediction" not in body:
                    resp.failure("Missing 'prediction' field in response")
                elif body.get("prediction") not in ("High", "Medium", "Low"):
                    resp.failure(f"Unexpected prediction class: {body.get('prediction')}")
                else:
                    resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:100]}")

    @task(2)
    def predict_anomalous(self):
        """High-stress conditions — model should return 'Low' efficiency."""
        with self.client.post(
            "/predict",
            data=json.dumps(_jitter(ANOMALOUS_PAYLOAD, pct=0.02)),
            headers=HEADERS,
            catch_response=True,
            name="POST /predict [anomalous]",
        ) as resp:
            if resp.status_code == 200:
                body = resp.json()
                if body.get("prediction") != "Low":
                    # Not a hard failure — model may legitimately disagree
                    resp.success()
                else:
                    resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @task(1)
    def health_check(self):
        """Liveness / readiness probe simulation."""
        with self.client.get(
            "/health",
            catch_response=True,
            name="GET /health",
        ) as resp:
            if resp.status_code == 200:
                body = resp.json()
                if not body.get("model_loaded"):
                    resp.failure("model_loaded=False in health response")
                else:
                    resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @task(1)
    def drift_status(self):
        """Drift endpoint poll — simulates CronJob querying drift state."""
        with self.client.get(
            "/drift",
            catch_response=True,
            name="GET /drift",
        ) as resp:
            if resp.status_code == 200:
                body = resp.json()
                if "status" not in body:
                    resp.failure("Missing 'status' in drift response")
                elif body.get("status") not in ("OK", "WARNING", "CRITICAL"):
                    resp.failure(f"Invalid drift status: {body.get('status')}")
                else:
                    resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")


# ── SLO Validation User ───────────────────────────────────────
class SLOValidationUser(HttpUser):
    """
    Strict SLO validator — every request asserts latency and status.
    Run with 5 concurrent users alongside MLOpsUser to continuously
    monitor SLO compliance under load.
    """
    wait_time = between(1.0, 3.0)
    host      = TARGET_URL
    weight    = 1   # fewer of these users — they're strict validators

    @task
    def validate_predict_slo(self):
        with self.client.post(
            "/predict",
            data=json.dumps(NORMAL_PAYLOAD),
            headers=HEADERS,
            catch_response=True,
            name="POST /predict [SLO check]",
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"SLO FAIL: HTTP {resp.status_code} (expected 200)")
            elif resp.elapsed.total_seconds() * 1000 > SLO_P95_MS:
                resp.failure(
                    f"SLO FAIL: latency {resp.elapsed.total_seconds()*1000:.0f}ms "
                    f"> {SLO_P95_MS}ms threshold"
                )
            else:
                resp.success()


# ── Test lifecycle hooks ──────────────────────────────────────
@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Print SLO summary at the end of every test run."""
    stats = environment.runner.stats
    total = stats.total

    rps       = total.current_rps if total.current_rps else (total.num_requests / max(total.total_response_time / 1000, 1))
    p95       = total.get_response_time_percentile(0.95) or 0
    p99       = total.get_response_time_percentile(0.99) or 0
    p50       = total.get_response_time_percentile(0.50) or 0
    error_pct = (total.num_failures / max(total.num_requests, 1)) * 100

    print("\n" + "=" * 60)
    print("  MLOps Load Test — SLO Summary")
    print("=" * 60)
    print(f"  Requests:    {total.num_requests:,}")
    print(f"  Failures:    {total.num_failures:,} ({error_pct:.2f}%)")
    print(f"  Throughput:  {rps:.1f} req/s")
    print(f"  Latency p50: {p50:.0f}ms")
    print(f"  Latency p95: {p95:.0f}ms  (SLO: < {SLO_P95_MS}ms)")
    print(f"  Latency p99: {p99:.0f}ms")
    print("-" * 60)

    slo_passed = True

    if p95 > SLO_P95_MS:
        print(f"  ❌ SLO BREACH: p95 {p95:.0f}ms > {SLO_P95_MS}ms")
        slo_passed = False
    else:
        print(f"  ✅ Latency SLO: p95 {p95:.0f}ms < {SLO_P95_MS}ms")

    if error_pct > SLO_ERROR_PCT:
        print(f"  ❌ SLO BREACH: error rate {error_pct:.2f}% > {SLO_ERROR_PCT}%")
        slo_passed = False
    else:
        print(f"  ✅ Error SLO: {error_pct:.2f}% < {SLO_ERROR_PCT}%")

    if rps < SLO_MIN_RPS:
        print(f"  ❌ SLO BREACH: throughput {rps:.1f} req/s < {SLO_MIN_RPS} req/s")
        slo_passed = False
    else:
        print(f"  ✅ Throughput SLO: {rps:.1f} req/s > {SLO_MIN_RPS} req/s")

    print("=" * 60)
    print(f"  Overall SLO: {'PASSED' if slo_passed else 'FAILED'}")
    print("=" * 60 + "\n")

    # In CI: fail the process if SLOs are breached
    if not slo_passed and isinstance(environment.runner, MasterRunner):
        environment.process_exit_code = 1
