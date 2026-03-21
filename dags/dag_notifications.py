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


SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK_URL", "")


def _send(payload: dict) -> None:
    """POST JSON payload to Slack webhook. Silently swallows errors so DAG never
    fails because of a notification issue."""
    try:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            SLACK_WEBHOOK,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as exc:
        print(f"[Slack] Notification failed (non-fatal): {exc}")


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

    _send({
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"✅ DAG Succeeded: {dag_id}"}
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*DAG:*\n`{dag_id}`"},
                    {"type": "mrkdwn", "text": f"*Status:*\n✅ SUCCESS"},
                    {"type": "mrkdwn", "text": f"*Duration:*\n{duration}"},
                    {"type": "mrkdwn", "text": f"*Run ID:*\n`{run_id[:40]}`"},
                ]
            },
            {
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": f"Completed at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"}]
            }
        ]
    })


def on_failure(context):
    """Called when a DAG run fails (any task failure)."""
    dag_id    = context["dag"].dag_id
    run_id    = context["dag_run"].run_id
    task_id   = context.get("task_instance", context.get("ti")).task_id
    exception = str(context.get("exception", "Unknown error"))[:200]
    duration  = _duration(context)

    _send({
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"🚨 DAG FAILED: {dag_id}"}
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*DAG:*\n`{dag_id}`"},
                    {"type": "mrkdwn", "text": f"*Status:*\n❌ FAILED"},
                    {"type": "mrkdwn", "text": f"*Failed Task:*\n`{task_id}`"},
                    {"type": "mrkdwn", "text": f"*Duration:*\n{duration}"},
                ]
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Error:*\n```{exception}```"}
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Run ID:* `{run_id[:60]}`"
                }
            },
            {
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": f"Failed at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} | Check Airflow UI for logs"}]
            }
        ]
    })


def on_retry(context):
    """Called when a task is retried."""
    dag_id   = context["dag"].dag_id
    task_id  = context.get("task_instance", context.get("ti")).task_id
    try_num  = context.get("task_instance", context.get("ti")).try_number
    exception = str(context.get("exception", "Unknown error"))[:150]

    _send({
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f":warning: *DAG Retry* | `{dag_id}` › `{task_id}`\n"
                        f"Attempt #{try_num} | Error: `{exception}`"
                    )
                }
            }
        ]
    })
