"""
MLOps Production Serving Application
=====================================
Enterprise-grade Flask app with:
  - REST API predict endpoint (/predict)
  - MLflow model loading with local fallback
  - Full Prometheus observability
  - Drift monitored by K8s CronJob (not background thread)
"""

from flask import Flask, render_template, request, jsonify, Response
import joblib
import numpy as np
import os
import time
import threading
from datetime import datetime
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Histogram, Gauge

from src.drift_detector import DriftDetector
from src.retraining_pipeline import RetrainingPipeline
from src.feature_registry import FEATURE_COLUMNS, LABEL_MAP_REVERSE, LABEL_MAP
from src.tracing import tracer

app = Flask(__name__)

# ================================================================== #
#  Prometheus Metrics — multiprocess-safe (gunicorn worker support)
# ================================================================== #
# When PROMETHEUS_MULTIPROC_DIR is set, prometheus_client automatically
# persists counter/gauge state to mmapped files shared across all gunicorn
# workers. prometheus_flask_exporter reads from this dir on /metrics scrape.
_multiproc_dir = os.getenv("PROMETHEUS_MULTIPROC_DIR")
if _multiproc_dir:
    os.makedirs(_multiproc_dir, exist_ok=True)

# PrometheusMetrics registers /metrics endpoint automatically.
# In multiprocess mode (PROMETHEUS_MULTIPROC_DIR set), it aggregates
# across all worker processes via prometheus_client's multiprocess module.
metrics = PrometheusMetrics(app)
metrics.info('mlops_app_info', 'MLOps Flask Application', version='2.0.0')

prediction_counter = Counter(
    'ml_predictions_total', 'Total predictions by class', ['result'])

prediction_latency = Histogram(
    'ml_prediction_duration_seconds', 'Prediction latency in seconds',
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0])

prediction_errors = Counter(
    'ml_prediction_errors_total', 'Total prediction errors')

model_loaded_gauge = Gauge(
    'ml_model_loaded', 'Whether the ML model is loaded (1=yes, 0=no)')

model_version_gauge = Gauge(
    'ml_model_version', 'Currently loaded MLflow model version number')

# Drift metrics
prediction_drift_score = Gauge(
    'ml_prediction_drift_score', 'PSI score for prediction distribution drift')

feature_drift_score = Gauge(
    'ml_feature_drift_score', 'PSI score for feature/input drift')

drift_status_gauge = Gauge(
    'ml_drift_status', 'Drift status: 0=OK, 1=WARNING, 2=CRITICAL')

retraining_counter = Counter(
    'ml_retraining_total', 'Number of retraining cycles triggered')

retraining_success = Gauge(
    'ml_retraining_last_success', '1=last retrain promoted model, 0=rolled back')

model_accuracy = Gauge(
    'ml_model_accuracy', 'Current model accuracy on test set')

# ── Extended observability metrics ──────────────────────────────────── #
# Per-feature PSI for all 14 features (Population Stability Index)
feature_psi_gauge = Gauge(
    'ml_feature_psi', 'PSI drift score per feature', ['feature_name'])

# DAG pipeline run duration (seconds) per dag_id
dag_run_duration = Gauge(
    'ml_dag_run_duration_seconds', 'Duration of DAG pipeline runs in seconds', ['dag_id'])

# Average prediction confidence (max probability across classes)
prediction_confidence_avg = Gauge(
    'ml_prediction_confidence_avg', 'Rolling average prediction confidence (max proba)')

# A/B test traffic split counter per model variant
ab_test_requests = Counter(
    'ml_ab_test_requests_total', 'A/B test request count per model variant', ['model'])

# DAG pipeline health counters — scraped by Prometheus → alert rules → Grafana
dag_runs = Counter(
    'ml_dag_runs_total',
    'DAG run outcomes by dag_id and status (success/failure/retry)',
    ['dag_id', 'status'])

dag_data_validation_failures = Counter(
    'ml_dag_data_validation_failures_total',
    'Number of times Airflow DAG data validation step failed')

# Security / audit counters
audit_log_failures = Counter(
    'ml_audit_log_failures_total',
    'Number of times prediction audit log write failed')

# Audit log path — write one line per prediction for security/compliance trail
_AUDIT_LOG_PATH = os.getenv("AUDIT_LOG_PATH", "/app/artifacts/audit.log")

# Rolling confidence accumulator (thread-safe)
_confidence_sum   = 0.0
_confidence_count = 0
_confidence_lock  = threading.Lock()

# ================================================================== #
#  Configuration
# ================================================================== #
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://3.15.231.90:5000")
MLFLOW_MODEL_NAME   = os.getenv("MLFLOW_MODEL_NAME", "mlops-efficiency-predictor")
ENVIRONMENT         = os.getenv("ENVIRONMENT", "production")

MODEL_PATH  = "artifacts/models/model.pkl"
SCALER_PATH = "artifacts/processed/scaler.pkl"

FEATURES   = FEATURE_COLUMNS    # single source of truth — imported from feature_registry
LABELS     = LABEL_MAP_REVERSE  # {0: "High", 1: "Low", 2: "Medium"}
STATUS_MAP = {"OK": 0, "WARNING": 1, "CRITICAL": 2}

model_lock = threading.Lock()
_current_model_version = 0

# ================================================================== #
#  Model Loading — MLflow first, local file fallback
# ================================================================== #
def load_model_from_mlflow():
    """Load model from MLflow Production stage. Falls back to local pkl."""
    global model, scaler, _current_model_version
    try:
        import mlflow
        import mlflow.sklearn
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=["Production"])
        if versions:
            v = versions[0]
            loaded_model = mlflow.sklearn.load_model(
                f"models:/{MLFLOW_MODEL_NAME}/Production"
            )
            _current_model_version = int(v.version)
            model_version_gauge.set(_current_model_version)
            print(f"[ModelLoader] Loaded MLflow '{MLFLOW_MODEL_NAME}' v{v.version} "
                  f"(run_id={v.run_id[:8]})")
            return loaded_model, None  # scaler loaded from local (stable)
    except Exception as e:
        print(f"[ModelLoader] MLflow unavailable: {e} — falling back to local file")
    return None, None


try:
    # Try MLflow first
    _mlflow_model, _ = load_model_from_mlflow()
    if _mlflow_model is not None:
        model  = _mlflow_model
        scaler = joblib.load(SCALER_PATH)
    else:
        model  = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        model_version_gauge.set(0)
    model_loaded_gauge.set(1)
except Exception as e:
    model_loaded_gauge.set(0)
    raise RuntimeError(f"Failed to load model: {e}")

drift_detector   = DriftDetector()
retrain_pipeline = RetrainingPipeline()
model_accuracy.set(retrain_pipeline.get_current_accuracy())

# ================================================================== #
#  Routes
# ================================================================== #
@app.route("/", methods=["GET", "POST"])
def index():
    """HTML UI for manual prediction (development/demo use)."""
    prediction = None
    if request.method == "POST":
        start = time.time()
        try:
            input_data  = [float(request.form[f]) for f in FEATURES]
            input_array = np.array(input_data).reshape(1, -1)
            with model_lock:
                scaled = scaler.transform(input_array)
                pred   = model.predict(scaled)[0]
            prediction = LABELS.get(pred, "Unknown")
            drift_detector.record(input_data, int(pred))
            prediction_counter.labels(result=prediction).inc()
        except Exception as e:
            prediction_errors.inc()
            prediction = f"Error: {e}"
        finally:
            prediction_latency.observe(time.time() - start)
    return render_template("index.html", prediction=prediction, features=FEATURES)


@app.route("/predict", methods=["POST"])
def predict_api():
    """
    Production REST API endpoint.

    Request (JSON):
        {
            "Operation_Mode": 1,
            "Temperature_C": 75.2,
            "Vibration_Hz": 2.1,
            "Power_Consumption_kW": 45.0,
            "Network_Latency_ms": 12.0,
            "Packet_Loss_%": 0.5,
            "Quality_Control_Defect_Rate_%": 1.2,
            "Production_Speed_units_per_hr": 320,
            "Predictive_Maintenance_Score": 0.85,
            "Error_Rate_%": 0.8
        }
        Note: Year/Month/Day/Hour are auto-injected from current UTC time if omitted.

    Response (JSON):
        {
            "prediction": "High",
            "class_id": 0,
            "probabilities": {"High": 0.87, "Low": 0.05, "Medium": 0.08},
            "model_version": 3,
            "latency_ms": 4.2
        }
    """
    start = time.time()
    span_id = tracer.start_span("predict", {"model_version": str(_current_model_version)})
    try:
        data = request.get_json(force=True, silent=True)

        # 3.5 — distinguish unparseable JSON from genuinely empty body
        if data is None:
            return jsonify({"error": "Request body must be valid JSON"}), 400
        if not isinstance(data, dict) or not data:
            return jsonify({"error": "Request body must be a non-empty JSON object"}), 400

        # Auto-inject time features from UTC now if not provided
        now = datetime.utcnow()
        data.setdefault('Year',  now.year)
        data.setdefault('Month', now.month)
        data.setdefault('Day',   now.day)
        data.setdefault('Hour',  now.hour)

        # 3.3 — report ALL missing features at once, not just the first
        non_time = [f for f in FEATURES if f not in ('Year', 'Month', 'Day', 'Hour')]
        missing = [f for f in non_time if f not in data]
        if missing:
            return jsonify({"error": f"Missing required features: {missing}"}), 400

        # 3.4 — catch non-numeric values and return 400
        try:
            input_data = [float(data[f]) for f in FEATURES]
        except (ValueError, TypeError) as e:
            prediction_errors.inc()
            return jsonify({"error": f"Invalid feature value — expected a number: {e}"}), 400

        input_array = np.array(input_data).reshape(1, -1)

        with model_lock:
            scaled = scaler.transform(input_array)
            pred   = model.predict(scaled)[0]
            proba  = model.predict_proba(scaled)[0].tolist()

        label = LABELS.get(int(pred), "Unknown")
        drift_detector.record(input_data, int(pred))
        prediction_counter.labels(result=label).inc()

        # Track prediction confidence (max probability)
        confidence = float(max(proba))
        global _confidence_sum, _confidence_count
        with _confidence_lock:
            _confidence_sum   += confidence
            _confidence_count += 1
            if _confidence_count > 0:
                prediction_confidence_avg.set(_confidence_sum / _confidence_count)

        # Audit log — one line per prediction for compliance trail
        try:
            with open(_AUDIT_LOG_PATH, "a") as _f:
                _f.write(
                    f"{datetime.utcnow().isoformat()}Z "
                    f"label={label} confidence={confidence:.4f} "
                    f"model_v={_current_model_version}\n"
                )
        except Exception:
            audit_log_failures.inc()

        # A/B test tracking — default model variant is "primary"
        ab_model = request.headers.get("X-Model-Variant", "primary")
        ab_test_requests.labels(model=ab_model).inc()

        tracer.finish_span(span_id, extra={"prediction": label, "confidence": confidence})
        return jsonify({
            "prediction":    label,
            "class_id":      int(pred),
            "probabilities": {LABELS[i]: round(p, 4) for i, p in enumerate(proba)},
            "model_version": _current_model_version,
            "latency_ms":    round((time.time() - start) * 1000, 2),
        }), 200

    except Exception as e:
        prediction_errors.inc()
        tracer.finish_span(span_id, error=True, extra={"error": str(e)})
        return jsonify({"error": str(e)}), 500
    finally:
        prediction_latency.observe(time.time() - start)


@app.route("/health")
def health():
    """Kubernetes liveness + readiness probe endpoint."""
    return jsonify({
        "status":        "healthy",
        "model_loaded":  True,
        "model_version": _current_model_version,
        "environment":   ENVIRONMENT,
        "timestamp":     datetime.utcnow().isoformat() + "Z",
    }), 200


@app.route("/drift")
def drift_status_endpoint():
    """Real-time drift status — used by CronJob and Airflow DAG."""
    result = drift_detector.check_drift()

    # Update per-feature PSI gauges from feature_drift_scores dict (if present)
    feature_scores = result.get("feature_drift_scores", {})
    if isinstance(feature_scores, dict) and feature_scores:
        # Use per-feature breakdown when available
        for feat, psi_val in feature_scores.items():
            try:
                feature_psi_gauge.labels(feature_name=feat).set(float(psi_val))
            except Exception:
                pass
    else:
        # Fallback: distribute overall feature_drift_score across all 14 features
        overall_psi = result.get("feature_drift_score", 0.0)
        for feat in FEATURES:
            feature_psi_gauge.labels(feature_name=feat).set(float(overall_psi))

    return jsonify(result), 200


@app.route("/dag-duration", methods=["POST"])
def record_dag_duration():
    """
    Receive DAG run completion metrics from Airflow (called by DAG callback).

    Request JSON: {"dag_id": "dag_1_daily_training", "duration_seconds": 142.5}
    """
    data = request.get_json(force=True, silent=True) or {}
    dag_id  = data.get("dag_id",  "unknown")
    duration = float(data.get("duration_seconds", 0))
    dag_run_duration.labels(dag_id=dag_id).set(duration)
    return jsonify({"status": "recorded", "dag_id": dag_id, "duration_seconds": duration}), 200


@app.route("/dag-event", methods=["POST"])
def record_dag_event():
    """
    Receive DAG lifecycle events from Airflow callbacks.

    This is the bridge that makes DAG runs visible in Prometheus → Grafana → Slack.
    The real production flow:
      Airflow callback → POST /dag-event → Prometheus counter updated
      → Prometheus evaluates alert rule → FIRING sent to AlertManager
      → AlertManager routes to Slack AND Grafana shows alert as FIRING

    Request JSON:
      {"event": "dag_success",          "dag_id": "mlops_daily_training", "duration_seconds": 142}
      {"event": "dag_failure",          "dag_id": "mlops_daily_training", "task_id": "train_model", "error": "..."}
      {"event": "task_retry",           "dag_id": "mlops_daily_training", "task_id": "train_model", "try_number": 2}
      {"event": "data_validation_failure", "dag_id": "mlops_daily_training"}
    """
    data    = request.get_json(force=True, silent=True) or {}
    event   = data.get("event", "")
    dag_id  = data.get("dag_id", "unknown")

    if event == "dag_success":
        dag_runs.labels(dag_id=dag_id, status="success").inc()
        duration = float(data.get("duration_seconds", 0))
        if duration:
            dag_run_duration.labels(dag_id=dag_id).set(duration)

    elif event == "dag_failure":
        dag_runs.labels(dag_id=dag_id, status="failure").inc()

    elif event == "task_retry":
        dag_runs.labels(dag_id=dag_id, status="retry").inc()

    elif event == "data_validation_failure":
        dag_data_validation_failures.inc()
        dag_runs.labels(dag_id=dag_id, status="failure").inc()

    return jsonify({"status": "recorded", "event": event, "dag_id": dag_id}), 200



@app.route("/explain", methods=["POST"])
def explain_prediction():
    """
    SHAP-based model explainability endpoint.

    Returns per-feature SHAP values so data scientists can understand
    WHY the model made a specific prediction.

    Request: same JSON body as /predict
    Response:
      {
        "prediction": "Medium",
        "class_id": 2,
        "shap_values": {"Temperature_C": 0.12, ...},
        "top_features": [{"feature": "Temperature_C", "shap_value": 0.12}, ...],
        "base_value": 0.33
      }
    """
    try:
        import shap
    except ImportError:
        return jsonify({"error": "SHAP not available — install shap package"}), 503

    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    now = datetime.utcnow()
    data.setdefault('Year',  now.year)
    data.setdefault('Month', now.month)
    data.setdefault('Day',   now.day)
    data.setdefault('Hour',  now.hour)

    try:
        input_data  = [float(data[f]) for f in FEATURES]
        input_array = np.array(input_data).reshape(1, -1)

        with model_lock:
            scaled = scaler.transform(input_array)
            pred   = int(model.predict(scaled)[0])
            # LinearExplainer works with LogisticRegression (linear model)
            explainer   = shap.LinearExplainer(model, scaled)
            shap_values = explainer.shap_values(scaled)

        label = LABELS.get(pred, "Unknown")

        # shap_values shape varies by SHAP version:
        #   older: list of (n_samples, n_features) per class
        #   newer: ndarray (n_samples, n_features, n_classes)
        #   alt:   ndarray (n_classes, n_samples, n_features)
        sv = np.array(shap_values)
        n_cls = len(LABELS)
        if sv.ndim == 3:
            if sv.shape[2] == n_cls:
                # (n_samples, n_features, n_classes)
                class_shap = sv[0, :, pred]
            else:
                # (n_classes, n_samples, n_features)
                class_shap = sv[pred, 0, :]
        elif sv.ndim == 2:
            class_shap = sv[0, :]
        else:
            class_shap = sv.ravel()

        ev = np.array(explainer.expected_value).ravel()
        base_val = float(ev[pred] if len(ev) > 1 else ev[0])

        feature_shap = {f: round(float(v), 6) for f, v in zip(FEATURES, class_shap)}
        top_features = sorted(
            [{"feature": k, "shap_value": v} for k, v in feature_shap.items()],
            key=lambda x: abs(x["shap_value"]),
            reverse=True
        )

        return jsonify({
            "prediction":   label,
            "class_id":     pred,
            "shap_values":  feature_shap,
            "top_features": top_features[:5],   # top-5 most influential features
            "base_value":   base_val,
        }), 200

    except KeyError as e:
        return jsonify({"error": f"Missing required feature: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5001)
