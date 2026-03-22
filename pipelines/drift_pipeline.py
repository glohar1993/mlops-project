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

import argparse
import requests
import json
import time
import os
import boto3
from datetime import datetime

# Config — uses Kubernetes service DNS internally
SERVING_APP_URL   = os.getenv("SERVING_APP_URL",   "http://flask-service.default.svc.cluster.local:80")
PSI_WARNING       = float(os.getenv("PSI_WARNING",  "0.1"))
PSI_CRITICAL      = float(os.getenv("PSI_CRITICAL", "0.2"))
MIN_WINDOW_SIZE   = int(os.getenv("MIN_WINDOW_SIZE", "10"))
REPORT_PATH       = os.getenv("REPORT_PATH", "/tmp/drift_report.json")

# Airflow REST API — triggers DAG 2 instead of calling /retrain directly
# This gives full DAG orchestration: backup → retrain → compare → promote/rollback
AIRFLOW_API_URL   = os.getenv("AIRFLOW_API_URL",  "")
AIRFLOW_USER      = os.getenv("AIRFLOW_USER",     "admin")
AIRFLOW_PASSWORD  = os.getenv("AIRFLOW_PASSWORD", "")

# S3 for durable drift reports (survives pod restarts)
REPORT_S3_BUCKET  = os.getenv("REPORT_S3_BUCKET", "")
REPORT_S3_PREFIX  = os.getenv("REPORT_S3_KEY_PREFIX", "drift-reports/")


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


def trigger_airflow_dag(drift_score: float, drift_status: str) -> dict:
    """
    Trigger Airflow DAG 2 via REST API.
    This is the production pattern: CronJob → Airflow DAG (with full orchestration).
    Previously called /retrain directly, which bypassed the backup/compare/promote logic.
    """
    if not AIRFLOW_API_URL:
        log("AIRFLOW_API_URL not set — falling back to direct /retrain call")
        return trigger_direct_retrain()

    dag_run_url = f"{AIRFLOW_API_URL}/api/v1/dags/mlops_drift_retraining/dagRuns"
    payload = {
        "conf": {
            "drift_score":  drift_score,
            "drift_status": drift_status,
            "triggered_by": "drift-detection-cronjob",
        }
    }
    log(f"Triggering Airflow DAG 2 via {dag_run_url}")
    r = requests.post(
        dag_run_url,
        json=payload,
        auth=(AIRFLOW_USER, AIRFLOW_PASSWORD),
        timeout=15,
    )
    r.raise_for_status()
    result = r.json()
    log(f"  DAG run triggered: {result.get('dag_run_id')} state={result.get('state')}")
    return {"dag_triggered": True, "dag_run_id": result.get("dag_run_id")}


def trigger_direct_retrain() -> dict:
    """Fallback: call Flask /retrain directly (used if Airflow is unavailable)."""
    log("Fallback: calling /retrain directly on serving pod...")
    r = requests.post(f"{SERVING_APP_URL}/retrain", timeout=120)
    r.raise_for_status()
    return r.json()


def save_report(report: dict):
    """Write drift report locally and upload to S3 for durability."""
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    log(f"Report saved locally to {REPORT_PATH}")

    if REPORT_S3_BUCKET:
        try:
            s3_key = f"{REPORT_S3_PREFIX}{datetime.utcnow().strftime('%Y/%m/%d/%H%M%S')}_drift_report.json"
            s3 = boto3.client("s3")
            s3.put_object(
                Bucket=REPORT_S3_BUCKET,
                Key=s3_key,
                Body=json.dumps(report, indent=2),
                ContentType="application/json",
            )
            log(f"Report uploaded to s3://{REPORT_S3_BUCKET}/{s3_key}")
        except Exception as e:
            log(f"Warning: S3 upload failed: {e} — local report still saved")


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
        log("Step 4: Triggering Airflow DAG 2 (full retrain + promote/rollback pipeline)...")
        report["status"] = "critical"

        # Airflow DAG provides: backup → retrain → compare → promote/rollback
        # This is more robust than calling /retrain directly
        trigger_result = trigger_airflow_dag(
            drift_score=overall_psi,
            drift_status="CRITICAL"
        )
        report["trigger_result"] = trigger_result

        if trigger_result.get("dag_triggered"):
            log(f"  ✅ Airflow DAG 2 triggered: {trigger_result.get('dag_run_id')}")
            report["action"] = "airflow-dag-triggered"
        elif trigger_result.get("model_replaced"):
            log(f"  ✅ Direct retrain deployed (Airflow fallback)")
            report["action"] = "retrained-and-deployed"
        else:
            log(f"  ↩️  Retrain rolled back or failed")
            report["action"] = "retrain-failed"

    # Step 4: Save report
    save_report(report)

    log("=" * 60)
    log(f"PIPELINE 3: COMPLETED — Status: {report['status']} | Action: {report['action']}")
    log("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MLOps Drift Detection Pipeline — queries /drift endpoint, "
                    "evaluates PSI, triggers Airflow DAG 2 on CRITICAL drift."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Check config and connectivity only, do not trigger retraining")
    parser.add_argument("--serving-url", default=None,
                        help=f"Override SERVING_APP_URL (default: {SERVING_APP_URL})")
    args = parser.parse_args()

    if args.serving_url:
        SERVING_APP_URL = args.serving_url

    if args.dry_run:
        print(f"[DRY RUN] SERVING_APP_URL : {SERVING_APP_URL}")
        print(f"[DRY RUN] PSI_WARNING     : {PSI_WARNING}")
        print(f"[DRY RUN] PSI_CRITICAL    : {PSI_CRITICAL}")
        print(f"[DRY RUN] AIRFLOW_API_URL : {AIRFLOW_API_URL or '(not set)'}")
        print("[DRY RUN] Config OK — exiting without running pipeline")
    else:
        run()
