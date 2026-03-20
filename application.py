"""
MLOps Production Serving Application
=====================================
Enterprise-grade Flask app with:
  - REST API predict endpoint (/predict)
  - Background drift monitoring (30s loop)
  - Model hot-swap (zero restart retraining)
  - MLflow model loading with local fallback
  - Full Prometheus observability
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
#  Background Drift Monitor (every 30 seconds)
# ================================================================== #
retraining_in_progress = False


def drift_monitor_loop():
    global model, scaler, retraining_in_progress

    while True:
        time.sleep(30)
        try:
            drift_result = drift_detector.check_drift()

            prediction_drift_score.set(drift_result["prediction_drift_score"])
            feature_drift_score.set(drift_result["feature_drift_score"])
            drift_status_gauge.set(STATUS_MAP.get(drift_result["status"], 0))

            print(f"[DriftMonitor] Status={drift_result['status']} "
                  f"PredDrift={drift_result['prediction_drift_score']} "
                  f"FeatDrift={drift_result['feature_drift_score']}")

            if drift_result["drift_detected"] and not retraining_in_progress:
                retraining_in_progress = True
                print("[DriftMonitor] CRITICAL drift — triggering retraining...")
                retraining_counter.inc()

                retrain_result = retrain_pipeline.retrain()

                if retrain_result["model_replaced"]:
                    # Try to load fresh model from MLflow, fallback to local
                    _mlflow_model, _ = load_model_from_mlflow()
                    with model_lock:
                        model  = _mlflow_model if _mlflow_model else joblib.load(MODEL_PATH)
                        scaler = joblib.load(SCALER_PATH)
                    retraining_success.set(1)
                    model_accuracy.set(retrain_result["new_accuracy"])
                    print(f"[DriftMonitor] Model hot-swapped! "
                          f"Accuracy: {retrain_result['new_accuracy']:.2%} "
                          f"MLflow v{_current_model_version}")
                    drift_detector.recent_predictions.clear()
                    drift_detector.recent_inputs.clear()
                else:
                    retraining_success.set(0)
                    print(f"[DriftMonitor] Rolled back: {retrain_result['reason']}")

                retraining_in_progress = False

        except Exception as e:
            print(f"[DriftMonitor] Error: {e}")
            retraining_in_progress = False


monitor_thread = threading.Thread(target=drift_monitor_loop, daemon=True)
monitor_thread.start()

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

        return jsonify({
            "prediction":    label,
            "class_id":      int(pred),
            "probabilities": {LABELS[i]: round(p, 4) for i, p in enumerate(proba)},
            "model_version": _current_model_version,
            "latency_ms":    round((time.time() - start) * 1000, 2),
        }), 200

    except Exception as e:
        prediction_errors.inc()
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
    return jsonify(result), 200


@app.route("/retrain", methods=["POST"])
def manual_retrain():
    """
    Trigger retraining and model hot-swap.
    Called by:
      - Airflow deploy_to_eks task (after DAG promotes new MLflow model)
      - CronJob drift pipeline on CRITICAL drift
      - Manual testing
    """
    global model, scaler
    logs = []
    retrain_result = retrain_pipeline.retrain(log_callback=logs.append)

    if retrain_result["model_replaced"]:
        # Prefer MLflow version, fallback to local file
        _mlflow_model, _ = load_model_from_mlflow()
        with model_lock:
            model  = _mlflow_model if _mlflow_model else joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
        model_accuracy.set(retrain_result["new_accuracy"])
        retraining_success.set(1)
    else:
        retraining_success.set(0)

    retraining_counter.inc()

    return jsonify({
        **retrain_result,
        "model_version": _current_model_version,
        "logs":          logs,
    }), 200


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
