"""
PIPELINE 3: Drift Detection Pipeline
=====================================
Runs independently as a Kubernetes CronJob every 2 minutes.
Responsibilities:
  - Query the serving app's /drift endpoint
  - Evaluate PSI score
  - If CRITICAL → call /retrain to trigger Pipeline 4
  - Log everything for Prometheus/Grafana

In production this would:
  - Query MLflow for model metadata
  - Send alerts to Slack/PagerDuty
  - Write drift report to S3/GCS
  - Trigger a separate Argo Workflow / Kubeflow pipeline
"""

import requests
import json
import time
import os
from datetime import datetime

# Config — uses Kubernetes service DNS internally
SERVING_APP_URL   = os.getenv("SERVING_APP_URL",   "http://flask-service:5001")
PSI_WARNING       = float(os.getenv("PSI_WARNING",  "0.1"))
PSI_CRITICAL      = float(os.getenv("PSI_CRITICAL", "0.2"))
MIN_WINDOW_SIZE   = int(os.getenv("MIN_WINDOW_SIZE", "10"))
REPORT_PATH       = os.getenv("REPORT_PATH", "/tmp/drift_report.json")


def log(msg):
    print(f"[{datetime.utcnow().isoformat()}] [DriftPipeline] {msg}", flush=True)


def check_app_health() -> bool:
    try:
        r = requests.get(f"{SERVING_APP_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception as e:
        log(f"Health check failed: {e}")
        return False


def get_drift_status() -> dict:
    r = requests.get(f"{SERVING_APP_URL}/drift", timeout=10)
    r.raise_for_status()
    return r.json()


def trigger_retraining() -> dict:
    log("Triggering retraining pipeline via /retrain endpoint...")
    r = requests.post(f"{SERVING_APP_URL}/retrain", timeout=120)
    r.raise_for_status()
    return r.json()


def save_report(report: dict):
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    log(f"Report saved to {REPORT_PATH}")


def run():
    log("=" * 60)
    log("PIPELINE 3: Drift Detection Pipeline — STARTED")
    log("=" * 60)

    report = {
        "pipeline":   "drift-detection",
        "timestamp":  datetime.utcnow().isoformat(),
        "status":     "unknown",
        "action":     "none",
        "drift_data": {},
        "retrain_result": None,
    }

    # Step 1: Check app is alive
    log("Step 1: Checking serving app health...")
    if not check_app_health():
        log("ERROR: Serving app is not healthy. Aborting.")
        report["status"] = "error"
        report["action"] = "aborted - app unhealthy"
        save_report(report)
        return

    log("Serving app is healthy ✅")

    # Step 2: Fetch drift metrics
    log("Step 2: Fetching drift metrics from serving app...")
    drift = get_drift_status()
    report["drift_data"] = drift

    log(f"  Prediction Drift PSI : {drift['prediction_drift_score']}")
    log(f"  Feature Drift Score  : {drift['feature_drift_score']}")
    log(f"  Overall Status       : {drift['status']}")
    log(f"  Window Size          : {drift['window_size']} predictions")
    log(f"  Prediction Dist      : {drift['current_pred_dist']}")

    # Step 3: Evaluate drift severity
    log("Step 3: Evaluating drift severity...")

    if drift["window_size"] < MIN_WINDOW_SIZE:
        log(f"  Not enough data yet ({drift['window_size']} < {MIN_WINDOW_SIZE}). Skipping.")
        report["status"] = "skipped"
        report["action"] = "insufficient data"
        save_report(report)
        return

    overall_psi = drift.get("overall_score", 0)

    if overall_psi < PSI_WARNING:
        log(f"  ✅ NO DRIFT — PSI {overall_psi:.4f} < {PSI_WARNING} (threshold)")
        report["status"] = "ok"
        report["action"] = "none"

    elif overall_psi < PSI_CRITICAL:
        log(f"  ⚠️  WARNING — PSI {overall_psi:.4f} >= {PSI_WARNING}. Monitoring.")
        report["status"] = "warning"
        report["action"] = "monitor"

    else:
        log(f"  🚨 CRITICAL DRIFT — PSI {overall_psi:.4f} >= {PSI_CRITICAL}!")
        log("Step 4: Triggering RETRAINING PIPELINE automatically...")
        report["status"] = "critical"

        retrain_result = trigger_retraining()
        report["retrain_result"] = retrain_result

        if retrain_result.get("model_replaced"):
            log(f"  ✅ Model retrained and DEPLOYED!")
            log(f"     Old Accuracy: {retrain_result['old_accuracy']:.2%}")
            log(f"     New Accuracy: {retrain_result['new_accuracy']:.2%}")
            report["action"] = "retrained-and-deployed"
        else:
            log(f"  ↩️  Retraining rolled back: {retrain_result.get('reason')}")
            report["action"] = "retrained-rolled-back"

    # Step 4: Save report
    save_report(report)

    log("=" * 60)
    log(f"PIPELINE 3: COMPLETED — Status: {report['status']} | Action: {report['action']}")
    log("=" * 60)


if __name__ == "__main__":
    run()
