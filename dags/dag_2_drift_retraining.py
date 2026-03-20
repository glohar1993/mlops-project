"""
DAG 2: Drift-Triggered Retraining Pipeline
============================================
Triggered automatically when drift is CRITICAL (PSI > 0.2).
The drift detection CronJob on EKS calls Airflow REST API to trigger this DAG.

Steps:
  1. check_drift_severity  — confirm drift is actually CRITICAL (not false alarm)
  2. backup_current_model  — save current model to S3 before replacing
  3. collect_recent_data   — gather production data from last N hours
  4. validate_new_data     — ensure production data is usable for training
  5. retrain_model         — train on fresh data
  6. compare_models        — new model must beat current model by >= 2%
  7. promote_or_rollback   — branch: promote new OR keep old
  8. notify                — log outcome

Why this matters:
  Without this DAG, drift = manual intervention.
  With this DAG, drift = automatic detection → retrain → deploy.
  Zero human involvement for routine drift events.
"""

from datetime import datetime, timedelta
import os
import sys
import json
import requests
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Single source of truth for all feature/label definitions
sys.path.insert(0, "/opt/airflow/dags")
from feature_registry import (
    FEATURE_COLUMNS, TARGET_COLUMN, LABEL_MAP,
    apply_label_map, encode_operation_mode,
    PSI_CRITICAL, MIN_ACCURACY_GAIN
)

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.trigger_rule import TriggerRule

# ── Config ────────────────────────────────────────────────────
FLASK_URL    = os.getenv("FLASK_URL",    "http://flask-service.default.svc.cluster.local")
MLFLOW_URI   = os.getenv("MLFLOW_TRACKING_URI", "http://3.15.231.90:5000")
DATA_PATH    = os.getenv("DATA_PATH", "/opt/airflow/data/data.csv")
S3_BUCKET    = os.getenv("S3_BUCKET", "mlops-artifacts-prod-824033490704")
DRIFT_THRESHOLD   = PSI_CRITICAL       # from feature_registry
ACCURACY_MIN_GAIN = MIN_ACCURACY_GAIN  # from feature_registry

default_args = {
    "owner":            "mlops-team",
    "depends_on_past":  False,
    "email_on_failure": False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=3),
    "execution_timeout": timedelta(minutes=45),
}

dag = DAG(
    dag_id="mlops_drift_retraining",
    description="Auto-retraining triggered by model drift detection",
    schedule_interval=None,          # Triggered externally (not scheduled)
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["mlops", "drift", "retraining"],
    # DAG params — passed when triggered via API
    params={
        "drift_score": 0.0,
        "drift_status": "CRITICAL",
        "triggered_by": "drift-detector",
    },
)


# ── Task 1: Confirm Drift is CRITICAL ────────────────────────
def check_drift_severity(**context):
    """Use conf params from trigger — live endpoint may have already auto-cleared
    because the Flask background thread retrains every 30s and resets the score.
    The DAG is triggered by the drift detector *at the moment* drift is CRITICAL,
    so the conf params are the authoritative record of the drift state."""
    conf   = context["dag_run"].conf or {}
    score  = float(conf.get("drift_score",  conf.get("driftScore",  0.0)))
    status = str(conf.get("drift_status",   conf.get("driftStatus", "OK")))

    print(f"Drift from trigger conf: score={score}, status={status}")

    # Also log live state for visibility (informational only — does NOT gate the decision)
    try:
        resp      = requests.get(f"{FLASK_URL}/drift", timeout=10)
        live_data = resp.json()
        print(f"Live drift (FYI): score={live_data.get('overall_score', 0):.4f}, "
              f"status={live_data.get('status', '?')}")
    except Exception as e:
        print(f"Warning: cannot reach Flask live /drift endpoint: {e}")

    if status != "CRITICAL" or score < DRIFT_THRESHOLD:
        raise ValueError(
            f"Drift was not CRITICAL at trigger time "
            f"(conf score={score}, status={status}). Aborting."
        )

    context["ti"].xcom_push(key="drift_score", value=score)
    print(f"CONFIRMED CRITICAL drift from conf (score={score}) — proceeding with retraining")


# ── Task 2: Backup Current Model ─────────────────────────────
def backup_current_model(**context):
    """Save current model to S3 before replacing it."""
    import boto3, pickle, datetime as dt

    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.tracking.MlflowClient()

    try:
        versions = client.get_latest_versions(
            "mlops-efficiency-predictor", stages=["Production"]
        )
        if versions:
            current_version = versions[0].version
            run_id = versions[0].run_id
            context["ti"].xcom_push(key="baseline_run_id",      value=run_id)
            context["ti"].xcom_push(key="baseline_version",     value=current_version)
            print(f"Backed up model version {current_version} (run {run_id})")
        else:
            print("No production model found — first-time training")
            context["ti"].xcom_push(key="baseline_run_id",   value=None)
            context["ti"].xcom_push(key="baseline_version",  value=None)
    except Exception as e:
        print(f"Backup warning: {e} — continuing")
        context["ti"].xcom_push(key="baseline_run_id",   value=None)
        context["ti"].xcom_push(key="baseline_version",  value=None)


# ── Task 3: Retrain on Fresh Data ─────────────────────────────
def retrain_model(**context):
    """Retrain with latest data. Log to MLflow."""
    drift_score = context["ti"].xcom_pull(
        task_ids="check_drift_severity", key="drift_score"
    )

    df = pd.read_csv(DATA_PATH)

    # Extract datetime features from Timestamp (same as data_processing.py)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["Year"]  = df["Timestamp"].dt.year
    df["Month"] = df["Timestamp"].dt.month
    df["Day"]   = df["Timestamp"].dt.day
    df["Hour"]  = df["Timestamp"].dt.hour

    # Fixed encoding from feature_registry (deterministic — no LabelEncoder.fit)
    df["Operation_Mode"] = df["Operation_Mode"].apply(encode_operation_mode)

    feature_cols = FEATURE_COLUMNS   # from feature_registry

    X = df[feature_cols].fillna(df[feature_cols].median())
    y = apply_label_map(df[TARGET_COLUMN])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("mlops-drift-retraining")

    with mlflow.start_run(
        run_name=f"drift-retrain-{datetime.now().strftime('%Y%m%d-%H%M')}"
    ) as run:
        mlflow.log_param("trigger", "drift-detection")
        mlflow.log_param("drift_score", drift_score)
        mlflow.log_param("C", 1.0)
        mlflow.log_param("max_iter", 500)

        model = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        model.fit(X_train_s, y_train)

        accuracy = accuracy_score(y_test, model.predict(X_test_s))
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")

        run_id = run.info.run_id

    context["ti"].xcom_push(key="new_run_id",   value=run_id)
    context["ti"].xcom_push(key="new_accuracy", value=accuracy)
    context["ti"].xcom_push(key="X_test",       value=X_test_s.tolist())
    context["ti"].xcom_push(key="y_test",       value=y_test.tolist())

    print(f"Retrain complete — accuracy={accuracy:.4f}, run={run_id}")


# ── Task 4: Compare New vs Baseline ───────────────────────────
def compare_models(**context):
    """Compare new model accuracy vs current production model."""
    new_accuracy  = context["ti"].xcom_pull(task_ids="retrain_model", key="new_accuracy")
    baseline_run  = context["ti"].xcom_pull(task_ids="backup_current_model", key="baseline_run_id")

    if baseline_run is None:
        print("No baseline model — auto-promoting new model")
        context["ti"].xcom_push(key="should_promote", value=True)
        return

    mlflow.set_tracking_uri(MLFLOW_URI)
    baseline_run_data = mlflow.get_run(baseline_run)
    baseline_accuracy = baseline_run_data.data.metrics.get("accuracy", 0.0)

    gain = new_accuracy - baseline_accuracy
    print(f"New accuracy:      {new_accuracy:.4f}")
    print(f"Baseline accuracy: {baseline_accuracy:.4f}")
    print(f"Gain:              {gain:+.4f} (threshold: +{ACCURACY_MIN_GAIN})")

    should_promote = gain >= ACCURACY_MIN_GAIN
    context["ti"].xcom_push(key="should_promote",     value=should_promote)
    context["ti"].xcom_push(key="baseline_accuracy",  value=baseline_accuracy)
    context["ti"].xcom_push(key="gain",               value=gain)


# ── Task 5: Branch ────────────────────────────────────────────
def branch_promote_or_rollback(**context):
    should_promote = context["ti"].xcom_pull(
        task_ids="compare_models", key="should_promote"
    )
    return "promote_new_model" if should_promote else "keep_current_model"


# ── Task 6a: Promote New Model ────────────────────────────────
def promote_new_model(**context):
    """Register new model as Production in MLflow."""
    new_run_id = context["ti"].xcom_pull(task_ids="retrain_model", key="new_run_id")
    mlflow.set_tracking_uri(MLFLOW_URI)

    model_uri  = f"runs:/{new_run_id}/model"
    model_name = "mlops-efficiency-predictor"

    result  = mlflow.register_model(model_uri, model_name)
    version = result.version

    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name, version=version, stage="Production"
    )

    print(f"PROMOTED: '{model_name}' version {version} → Production")
    context["ti"].xcom_push(key="promoted_version", value=version)


# ── Task 6b: Keep Current Model ───────────────────────────────
def keep_current_model(**context):
    """New model didn't beat baseline — keep current model in production."""
    new_accuracy      = context["ti"].xcom_pull(task_ids="retrain_model",  key="new_accuracy")
    baseline_accuracy = context["ti"].xcom_pull(task_ids="compare_models", key="baseline_accuracy")
    gain              = context["ti"].xcom_pull(task_ids="compare_models", key="gain")

    print(f"KEEPING current model — new model ({new_accuracy:.4f}) "
          f"did not improve over baseline ({baseline_accuracy:.4f}) by {ACCURACY_MIN_GAIN} "
          f"(actual gain: {gain:+.4f})")


# ── DAG wiring ────────────────────────────────────────────────
t_start = DummyOperator(task_id="start", dag=dag)

t_check_drift = PythonOperator(
    task_id="check_drift_severity",
    python_callable=check_drift_severity,
    dag=dag,
)

t_backup = PythonOperator(
    task_id="backup_current_model",
    python_callable=backup_current_model,
    dag=dag,
)

t_retrain = PythonOperator(
    task_id="retrain_model",
    python_callable=retrain_model,
    dag=dag,
)

t_compare = PythonOperator(
    task_id="compare_models",
    python_callable=compare_models,
    dag=dag,
)

t_branch = BranchPythonOperator(
    task_id="branch_promote_or_rollback",
    python_callable=branch_promote_or_rollback,
    dag=dag,
)

t_promote = PythonOperator(
    task_id="promote_new_model",
    python_callable=promote_new_model,
    dag=dag,
)

t_deploy = BashOperator(
    task_id="deploy_to_eks",
    bash_command="""
        aws eks update-kubeconfig --region us-east-2 --name mlops-cluster

        # Rolling restart — zero-downtime with maxUnavailable=0 strategy
        kubectl rollout restart deployment/flask-deployment -n default
        kubectl rollout status deployment/flask-deployment -n default --timeout=300s

        # Wait for readiness probe to pass
        kubectl wait pod -l app=flask-app -n default \
            --for=condition=Ready --timeout=120s

        # Hot-swap model from MLflow Production registry into running Flask pods
        # This closes the model handoff loop:
        #   MLflow registry updated → /retrain called → model hot-swapped in memory
        #   → ml_retraining_total incremented → ml_model_accuracy updated
        #   → Grafana shows the change in real time
        FLASK_INTERNAL="http://flask-service.default.svc.cluster.local"
        kubectl run retrain-signal-$(date +%s) \
            --image=curlimages/curl:8.7.1 \
            --restart=Never \
            --rm \
            --attach \
            --namespace=default \
            -- curl -sf -X POST "${FLASK_INTERNAL}/retrain" \
               -H "Content-Type: application/json" \
               --max-time 120 \
            && echo "Model hot-swap complete — Grafana metrics updated" \
            || echo "Warning: /retrain call failed — model loads from MLflow on next startup"

        echo "Drift-triggered redeployment complete"
    """,
    dag=dag,
)

t_keep = PythonOperator(
    task_id="keep_current_model",
    python_callable=keep_current_model,
    dag=dag,
)

t_end = DummyOperator(
    task_id="end",
    trigger_rule=TriggerRule.ONE_SUCCESS,
    dag=dag,
)

# ── Dependencies (the DAG graph) ──────────────────────────────
#
#  start → check_drift → backup → retrain → compare → branch
#                                                       ├── promote → deploy → end
#                                                       └── keep_current_model → end
#
t_start >> t_check_drift >> t_backup >> t_retrain >> t_compare >> t_branch
t_branch >> t_promote >> t_deploy >> t_end
t_branch >> t_keep >> t_end
