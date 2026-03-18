from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import time
import threading
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Histogram, Gauge

from src.drift_detector import DriftDetector
from src.retraining_pipeline import RetrainingPipeline

app = Flask(__name__)

# ------------------------------------------------------------------ #
#  Prometheus Metrics
# ------------------------------------------------------------------ #
metrics = PrometheusMetrics(app)
metrics.info('mlops_app_info', 'MLOps Flask Application', version='2.0.0')

prediction_counter = Counter(
    'ml_predictions_total', 'Total predictions', ['result'])

prediction_latency = Histogram(
    'ml_prediction_duration_seconds', 'Prediction latency',
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0])

prediction_errors = Counter(
    'ml_prediction_errors_total', 'Total prediction errors')

model_loaded = Gauge(
    'ml_model_loaded', 'Whether the ML model is loaded (1=yes)')

# Drift metrics
prediction_drift_score = Gauge(
    'ml_prediction_drift_score', 'PSI score for prediction distribution drift')

feature_drift_score = Gauge(
    'ml_feature_drift_score', 'PSI score for feature/input drift')

drift_status = Gauge(
    'ml_drift_status', 'Drift status: 0=OK, 1=WARNING, 2=CRITICAL')

retraining_counter = Counter(
    'ml_retraining_total', 'Number of retraining cycles triggered')

retraining_success = Gauge(
    'ml_retraining_last_success', '1=last retrain replaced model, 0=rolled back')

model_accuracy = Gauge(
    'ml_model_accuracy', 'Current model accuracy on test set')

# ------------------------------------------------------------------ #
#  Load Model & Initialize Drift Detector
# ------------------------------------------------------------------ #
MODEL_PATH  = "artifacts/models/model.pkl"
SCALER_PATH = "artifacts/processed/scaler.pkl"

model_lock = threading.Lock()

try:
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    model_loaded.set(1)
except Exception as e:
    model_loaded.set(0)
    raise e

drift_detector    = DriftDetector()
retrain_pipeline  = RetrainingPipeline()

# Set initial accuracy
model_accuracy.set(retrain_pipeline.get_current_accuracy())

FEATURES = [
    'Operation_Mode', 'Temperature_C', 'Vibration_Hz',
    'Power_Consumption_kW', 'Network_Latency_ms', 'Packet_Loss_%',
    'Quality_Control_Defect_Rate_%', 'Production_Speed_units_per_hr',
    'Predictive_Maintenance_Score', 'Error_Rate_%',
    'Year', 'Month', 'Day', 'Hour'
]

LABELS = {0: "High", 1: "Low", 2: "Medium"}
STATUS_MAP = {"OK": 0, "WARNING": 1, "CRITICAL": 2}

# ------------------------------------------------------------------ #
#  Background Drift Monitor (runs every 30 seconds)
# ------------------------------------------------------------------ #
retraining_in_progress = False

def drift_monitor_loop():
    global model, scaler, retraining_in_progress

    while True:
        time.sleep(30)
        try:
            drift_result = drift_detector.check_drift()

            prediction_drift_score.set(drift_result["prediction_drift_score"])
            feature_drift_score.set(drift_result["feature_drift_score"])
            drift_status.set(STATUS_MAP.get(drift_result["status"], 0))

            print(f"[DriftMonitor] Status={drift_result['status']} "
                  f"PredDrift={drift_result['prediction_drift_score']} "
                  f"FeatDrift={drift_result['feature_drift_score']}")

            # Auto-trigger retraining on CRITICAL drift
            if drift_result["drift_detected"] and not retraining_in_progress:
                retraining_in_progress = True
                print("[DriftMonitor] CRITICAL drift detected! Triggering retraining...")
                retraining_counter.inc()

                retrain_result = retrain_pipeline.retrain()

                if retrain_result["model_replaced"]:
                    # Hot-swap model in memory
                    with model_lock:
                        model  = joblib.load(MODEL_PATH)
                        scaler = joblib.load(SCALER_PATH)
                    retraining_success.set(1)
                    model_accuracy.set(retrain_result["new_accuracy"])
                    print(f"[DriftMonitor] Model hot-swapped! New accuracy: {retrain_result['new_accuracy']:.2%}")

                    # Reset drift window after retraining
                    drift_detector.recent_predictions.clear()
                    drift_detector.recent_inputs.clear()
                else:
                    retraining_success.set(0)
                    print(f"[DriftMonitor] Retraining rolled back: {retrain_result['reason']}")

                retraining_in_progress = False

        except Exception as e:
            print(f"[DriftMonitor] Error: {e}")
            retraining_in_progress = False


# Start background drift monitor thread
monitor_thread = threading.Thread(target=drift_monitor_loop, daemon=True)
monitor_thread.start()

# ------------------------------------------------------------------ #
#  Routes
# ------------------------------------------------------------------ #
@app.route("/", methods=["GET", "POST"])
def index():
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

            # Record for drift detection
            drift_detector.record(input_data, int(pred))
            prediction_counter.labels(result=prediction).inc()

        except Exception as e:
            prediction_errors.inc()
            prediction = f"Error : {e}"
        finally:
            prediction_latency.observe(time.time() - start)

    return render_template("index.html", prediction=prediction, features=FEATURES)


@app.route("/health")
def health():
    return jsonify({"status": "healthy", "model_loaded": True}), 200


@app.route("/drift")
def drift_status_endpoint():
    """Real-time drift status endpoint."""
    result = drift_detector.check_drift()
    return jsonify(result), 200


@app.route("/retrain", methods=["POST"])
def manual_retrain():
    """Manually trigger retraining (for testing)."""
    global model, scaler
    logs = []
    result = retrain_pipeline.retrain(log_callback=logs.append)
    if result["model_replaced"]:
        with model_lock:
            model  = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
        model_accuracy.set(result["new_accuracy"])
        retraining_success.set(1)
    else:
        retraining_success.set(0)
    retraining_counter.inc()
    return jsonify(result), 200


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5001)
