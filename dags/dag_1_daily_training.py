"""
DAG 1: Scheduled Daily Model Training
======================================
Runs every day at midnight.
Steps:
  1. validate_data       — check schema, nulls, ranges
  2. preprocess_data     — feature engineering, train/test split
  3. train_model         — Logistic Regression + MLflow logging
  4. evaluate_model      — accuracy, F1, confusion matrix
  5. register_model      — push to MLflow Model Registry if accuracy > threshold
  6. deploy_model        — rolling update on EKS if model approved
  7. notify              — log results

Why a DAG?
  - Each step depends on the previous (no step 3 if step 1 fails)
  - Full retry logic per task (not all-or-nothing)
  - Visual pipeline in Airflow UI — see exactly which step failed
  - Audit trail of every training run
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.trigger_rule import TriggerRule

import os
import sys

# feature_registry is pure Python (no ML deps) — safe to import at module level
sys.path.insert(0, "/opt/airflow/dags")
from feature_registry import (
    FEATURE_COLUMNS, TARGET_COLUMN, LABEL_MAP,
    apply_label_map, encode_operation_mode, MIN_ACCURACY
)

# ── Config ────────────────────────────────────────────────────
MLFLOW_URI        = os.getenv("MLFLOW_TRACKING_URI", "http://3.15.231.90:5000")
DATA_PATH         = os.getenv("DATA_PATH", "/opt/airflow/data/data.csv")
MODEL_ACCURACY_THRESHOLD = MIN_ACCURACY   # from feature_registry — single source of truth
ECR_IMAGE         = os.getenv("ECR_IMAGE", "824033490704.dkr.ecr.us-east-2.amazonaws.com/mlops-flask-app:latest")
EKS_NAMESPACE     = "default"

# ── Default task args ─────────────────────────────────────────
default_args = {
    "owner":            "mlops-team",
    "depends_on_past":  False,
    "email_on_failure": False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=30),
}

# ── DAG definition ────────────────────────────────────────────
dag = DAG(
    dag_id="mlops_daily_training",
    description="Daily MLOps model training pipeline",
    schedule_interval="0 0 * * *",          # Every day at midnight
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,                       # Never run two at the same time
    default_args=default_args,
    tags=["mlops", "training", "daily"],
)


# ── Task 1: Validate Data ─────────────────────────────────────
def validate_data(**context):
    """Check data quality before training."""
    import pandas as pd  # noqa: PLC0415 — lazy import: not in airflow base image
    print(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    required_cols = [
        "Operation_Mode", "Temperature_C", "Vibration_Hz",
        "Power_Consumption_kW", "Network_Latency_ms", "Packet_Loss_%",
        "Quality_Control_Defect_Rate_%", "Production_Speed_units_per_hr",
        "Predictive_Maintenance_Score", "Error_Rate_%",
        "Year", "Month", "Day", "Hour", "Efficiency_Category"
    ]

    errors = []

    # Schema check
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        errors.append(f"Missing columns: {missing}")

    # Null check
    null_pct = df.isnull().sum() / len(df) * 100
    high_null = null_pct[null_pct > 10].to_dict()
    if high_null:
        errors.append(f"High null %: {high_null}")

    # Size check
    if len(df) < 100:
        errors.append(f"Too few rows: {len(df)} (min 100)")

    if errors:
        raise ValueError(f"Data validation FAILED:\n" + "\n".join(errors))

    print(f"Data validation PASSED — {len(df)} rows, {len(df.columns)} cols")

    # Pass data stats to next task via XCom
    context["ti"].xcom_push(key="row_count", value=len(df))
    context["ti"].xcom_push(key="data_path", value=DATA_PATH)


# ── Task 2: Preprocess ────────────────────────────────────────
def preprocess_data(**context):
    """Feature engineering and train/test split."""
    import pandas as pd  # noqa: PLC0415
    from sklearn.model_selection import train_test_split  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415
    data_path = context["ti"].xcom_pull(task_ids="validate_data", key="data_path")
    df = pd.read_csv(data_path)

    feature_cols = FEATURE_COLUMNS   # from feature_registry

    X = df[feature_cols].fillna(df[feature_cols].median())
    y = apply_label_map(df[TARGET_COLUMN])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    import pickle, tempfile
    tmp = tempfile.mkdtemp()
    pd.DataFrame(X_train_scaled).to_parquet(f"{tmp}/X_train.parquet")
    pd.DataFrame(X_test_scaled).to_parquet(f"{tmp}/X_test.parquet")
    y_train.to_frame().to_parquet(f"{tmp}/y_train.parquet")
    y_test.to_frame().to_parquet(f"{tmp}/y_test.parquet")
    with open(f"{tmp}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    context["ti"].xcom_push(key="processed_dir", value=tmp)
    print(f"Preprocessed — train: {len(X_train)}, test: {len(X_test)}")


# ── Task 3: Train Model ───────────────────────────────────────
def train_model(**context):
    """Train Logistic Regression and log to MLflow."""
    import pandas as pd  # noqa: PLC0415
    import mlflow  # noqa: PLC0415
    import mlflow.sklearn  # noqa: PLC0415
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
    processed_dir = context["ti"].xcom_pull(
        task_ids="preprocess_data", key="processed_dir"
    )

    X_train = pd.read_parquet(f"{processed_dir}/X_train.parquet").values
    y_train = pd.read_parquet(f"{processed_dir}/y_train.parquet").values.ravel()

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("mlops-daily-training")

    with mlflow.start_run(run_name=f"dag-run-{datetime.now().strftime('%Y%m%d')}") as run:
        params = {"C": 1.0, "max_iter": 500, "solver": "lbfgs", "random_state": 42}
        mlflow.log_params(params)

        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        mlflow.sklearn.log_model(model, "model")
        run_id = run.info.run_id

    context["ti"].xcom_push(key="run_id", value=run_id)
    context["ti"].xcom_push(key="processed_dir", value=processed_dir)
    print(f"Training complete — MLflow run: {run_id}")


# ── Task 4: Evaluate Model ────────────────────────────────────
def evaluate_model(**context):
    """Evaluate model and decide whether to promote."""
    import pandas as pd  # noqa: PLC0415
    import mlflow  # noqa: PLC0415
    import mlflow.sklearn  # noqa: PLC0415
    from sklearn.metrics import accuracy_score, f1_score  # noqa: PLC0415
    processed_dir = context["ti"].xcom_pull(
        task_ids="train_model", key="processed_dir"
    )
    run_id = context["ti"].xcom_pull(task_ids="train_model", key="run_id")

    X_test = pd.read_parquet(f"{processed_dir}/X_test.parquet").values
    y_test = pd.read_parquet(f"{processed_dir}/y_test.parquet").values.ravel()

    mlflow.set_tracking_uri(MLFLOW_URI)

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1       = f1_score(y_test, y_pred, average="weighted")

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

    print(f"Accuracy: {accuracy:.4f} | F1: {f1:.4f} | Threshold: {MODEL_ACCURACY_THRESHOLD}")

    context["ti"].xcom_push(key="accuracy", value=accuracy)
    context["ti"].xcom_push(key="run_id",   value=run_id)

    return accuracy


# ── Task 5: Branch — promote or skip ─────────────────────────
def branch_on_accuracy(**context):
    """Decide next step based on accuracy threshold."""
    accuracy = context["ti"].xcom_pull(task_ids="evaluate_model")
    if accuracy >= MODEL_ACCURACY_THRESHOLD:
        print(f"Accuracy {accuracy:.4f} >= threshold {MODEL_ACCURACY_THRESHOLD} → PROMOTE")
        return "register_model"
    else:
        print(f"Accuracy {accuracy:.4f} < threshold {MODEL_ACCURACY_THRESHOLD} → SKIP DEPLOY")
        return "accuracy_too_low"


# ── Task 6: Register Model in MLflow Registry ─────────────────
def register_model(**context):
    """Register model in MLflow Model Registry."""
    import mlflow  # noqa: PLC0415
    run_id = context["ti"].xcom_pull(task_ids="evaluate_model", key="run_id")
    mlflow.set_tracking_uri(MLFLOW_URI)

    model_uri  = f"runs:/{run_id}/model"
    model_name = "mlops-efficiency-predictor"

    result = mlflow.register_model(model_uri, model_name)
    version = result.version

    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name, version=version, stage="Production"
    )

    print(f"Registered model '{model_name}' version {version} → Production")
    context["ti"].xcom_push(key="model_version", value=version)


# ── DAG wiring ────────────────────────────────────────────────
t_start    = DummyOperator(task_id="start", dag=dag)

t_validate = PythonOperator(
    task_id="validate_data",
    python_callable=validate_data,
    dag=dag,
)

t_preprocess = PythonOperator(
    task_id="preprocess_data",
    python_callable=preprocess_data,
    dag=dag,
)

t_train = PythonOperator(
    task_id="train_model",
    python_callable=train_model,
    dag=dag,
)

t_evaluate = PythonOperator(
    task_id="evaluate_model",
    python_callable=evaluate_model,
    dag=dag,
)

t_branch = BranchPythonOperator(
    task_id="branch_on_accuracy",
    python_callable=branch_on_accuracy,
    dag=dag,
)

t_register = PythonOperator(
    task_id="register_model",
    python_callable=register_model,
    dag=dag,
)

t_deploy = BashOperator(
    task_id="deploy_to_eks",
    bash_command="""
        aws eks update-kubeconfig --region us-east-2 --name mlops-cluster

        # Rolling restart — picks up new ECR image (zero-downtime with maxUnavailable=0)
        kubectl rollout restart deployment/flask-deployment -n default
        kubectl rollout status deployment/flask-deployment -n default --timeout=300s

        # Wait for pods to pass readiness probe
        kubectl wait pod -l app=flask-app -n default \
            --for=condition=Ready --timeout=120s

        # Signal Flask to hot-swap model from MLflow Production registry
        # This increments ml_retraining_total and updates ml_model_accuracy in Grafana
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
            && echo "Model hot-swap triggered — ml_retraining_total incremented" \
            || echo "Warning: /retrain call failed — model loads from MLflow on next startup"

        echo "DAG 1 deploy complete"
    """,
    dag=dag,
)

t_skip = DummyOperator(task_id="accuracy_too_low", dag=dag)

t_end = DummyOperator(
    task_id="end",
    trigger_rule=TriggerRule.ONE_SUCCESS,
    dag=dag,
)

# ── Dependencies (the DAG graph) ──────────────────────────────
#
#  start → validate → preprocess → train → evaluate → branch
#                                                       ├── register → deploy → end
#                                                       └── accuracy_too_low  → end
#
t_start >> t_validate >> t_preprocess >> t_train >> t_evaluate >> t_branch
t_branch >> t_register >> t_deploy >> t_end
t_branch >> t_skip >> t_end
