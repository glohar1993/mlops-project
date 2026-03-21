"""
dag_notifications.py
====================
Reusable Airflow DAG callbacks — production-correct flow.

REAL PRODUCTION FLOW (what Grafana actually sees):
  1. DAG run completes (success / failure / retry)
  2. Callback fires → POST /dag-event to Flask
  3. Flask increments ml_dag_runs_total{dag_id, status} counter
  4. Prometheus scrapes /metrics every 15s → sees counter rise
  5. Prometheus alert rule evaluates: rate(ml_dag_runs_total{status="failure"}[5m]) > 0
  6. Prometheus fires alert → sends to AlertManager
  7. AlertManager routes → Slack AND Grafana "Alerting" panel shows FIRING

Why NOT direct AlertManager push?
  Direct pushes (POST /api/v2/alerts) bypass Prometheus.
  Grafana only shows alerts that go through Prometheus rules.
  Direct pushes appear in AlertManager UI but NOT in Grafana.

Usage in any DAG:
    from dag_notifications import on_success, on_failure, on_retry

    dag = DAG(
        ...
        on_success_callback=on_success,
        on_failure_callback=on_failure,
    )
    default_args = {
        ...
        "on_retry_callback": on_retry,
    }
"""

import os
import urllib.request
import json
from datetime import datetime, timezone


_FLASK_URL = os.getenv(
    "FLASK_URL",
    "http://flask-service.default.svc.cluster.local",
)


def _post_flask_event(payload: dict) -> None:
    """POST DAG event to Flask /dag-event.
    Flask increments Prometheus counters → Prometheus fires alert rules →
    Grafana shows FIRING alert → AlertManager sends to Slack.
    Non-fatal: DAGs never fail due to notification issues."""
    try:
        data = json.dumps(payload).encode()
        req  = urllib.request.Request(
            f"{_FLASK_URL}/dag-event",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as exc:
        print(f"[Notify] Flask /dag-event unreachable (non-fatal): {exc}")


def _duration_secs(context) -> float:
    """Return DAG run duration in seconds."""
    try:
        start = context["dag_run"].start_date
        if start:
            now = datetime.now(timezone.utc)
            return (now - start).total_seconds()
    except Exception:
        pass
    return 0.0


def _duration_str(secs: float) -> str:
    m, s = divmod(int(secs), 60)
    return f"{m}m {s}s"


def on_success(context):
    """Called when a DAG run completes successfully.
    Posts to Flask → Prometheus counter → alert rule → Grafana + Slack."""
    dag_id   = context["dag"].dag_id
    secs     = _duration_secs(context)
    _post_flask_event({
        "event":            "dag_success",
        "dag_id":           dag_id,
        "duration_seconds": secs,
    })
    print(f"[Notify] DAG success recorded: {dag_id} ({_duration_str(secs)})")


def on_failure(context):
    """Called when a DAG run fails.
    Posts to Flask → Prometheus counter → alert rule → Grafana + Slack."""
    dag_id    = context["dag"].dag_id
    run_id    = context["dag_run"].run_id
    task_id   = context.get("task_instance", context.get("ti")).task_id
    exception = str(context.get("exception", "Unknown error"))[:200]
    secs      = _duration_secs(context)
    _post_flask_event({
        "event":   "dag_failure",
        "dag_id":  dag_id,
        "task_id": task_id,
        "error":   exception,
        "run_id":  run_id[:60],
    })
    print(f"[Notify] DAG failure recorded: {dag_id} task={task_id} error={exception[:60]}")


def on_retry(context):
    """Called when a task is retried.
    Posts to Flask → Prometheus counter → alert rule → Grafana + Slack."""
    dag_id  = context["dag"].dag_id
    task_id = context.get("task_instance", context.get("ti")).task_id
    try_num = context.get("task_instance", context.get("ti")).try_number
    _post_flask_event({
        "event":      "task_retry",
        "dag_id":     dag_id,
        "task_id":    task_id,
        "try_number": try_num,
    })
    print(f"[Notify] Task retry recorded: {dag_id} task={task_id} attempt={try_num}")
