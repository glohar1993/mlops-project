"""
dag_notifications.py
====================
Reusable Airflow DAG callbacks that post to Slack.

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


# AlertManager internal URL — reachable from any pod in the cluster.
# AlertManager forwards alerts to Slack via the configured webhook.
# Fallback: direct Slack webhook if set in SLACK_WEBHOOK_URL env var.
_ALERTMANAGER_URL = os.getenv(
    "ALERTMANAGER_URL",
    "http://alertmanager-service.default.svc.cluster.local:9093",
)
SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK_URL", "")


def _post_alert(alertname: str, severity: str, summary: str, description: str) -> None:
    """Send an alert to AlertManager (internal cluster URL).
    AlertManager forwards it to Slack via the configured webhook.
    Silently swallows errors so DAGs never fail due to notification issues."""
    try:
        from datetime import timezone
        payload = [{
            "labels": {
                "alertname": alertname,
                "severity": severity,
                "service":  "airflow",
                "source":   "dag-callback",
            },
            "annotations": {
                "summary":     summary,
                "description": description,
            },
            "startsAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }]
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{_ALERTMANAGER_URL}/api/v2/alerts",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as exc:
        print(f"[Notify] AlertManager unreachable (non-fatal): {exc}")


def _duration(context) -> str:
    """Return human-readable DAG run duration."""
    try:
        start = context["dag_run"].start_date
        if start:
            now = datetime.now(timezone.utc)
            secs = int((now - start).total_seconds())
            m, s = divmod(secs, 60)
            return f"{m}m {s}s"
    except Exception:
        pass
    return "?"


def on_success(context):
    """Called when a DAG run completes successfully."""
    dag_id   = context["dag"].dag_id
    run_id   = context["dag_run"].run_id
    duration = _duration(context)
    _post_alert(
        alertname=f"DagSucceeded",
        severity="info",
        summary=f"DAG succeeded: {dag_id} ({duration})",
        description=(
            f"DAG `{dag_id}` completed successfully in {duration}. "
            f"Run: `{run_id[:60]}`"
        ),
    )


def on_failure(context):
    """Called when a DAG run fails (any task failure)."""
    dag_id    = context["dag"].dag_id
    run_id    = context["dag_run"].run_id
    task_id   = context.get("task_instance", context.get("ti")).task_id
    exception = str(context.get("exception", "Unknown error"))[:200]
    duration  = _duration(context)
    _post_alert(
        alertname="DagFailed",
        severity="critical",
        summary=f"DAG FAILED: {dag_id} — task `{task_id}` ({duration})",
        description=(
            f"DAG `{dag_id}` failed at task `{task_id}` after {duration}. "
            f"Error: {exception}. "
            f"Run: `{run_id[:60]}`"
        ),
    )


def on_retry(context):
    """Called when a task is retried."""
    dag_id    = context["dag"].dag_id
    task_id   = context.get("task_instance", context.get("ti")).task_id
    try_num   = context.get("task_instance", context.get("ti")).try_number
    exception = str(context.get("exception", "Unknown error"))[:150]
    _post_alert(
        alertname="DagTaskRetry",
        severity="warning",
        summary=f"DAG retry: {dag_id} › {task_id} (attempt #{try_num})",
        description=(
            f"Task `{task_id}` in DAG `{dag_id}` is retrying (attempt #{try_num}). "
            f"Error: {exception}"
        ),
    )
