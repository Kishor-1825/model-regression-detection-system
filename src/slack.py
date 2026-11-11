from __future__ import annotations

import logging
import os
from typing import Optional
import requests

from src.models import DiffResult, RunSummary

logger = logging.getLogger("slack_notifier")


def send_slack_alert(
    diff: DiffResult,
    candidate_summary: RunSummary,
    baseline_summary: RunSummary,
    report_url: Optional[str] = None,
    webhook_url: Optional[str] = None
) -> bool:
    """Send a structured Slack alert via Incoming Webhook.
    Returns True if sent successfully, False otherwise.
    """
    url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
    if not url or "00000000" in url or "xxxx" in url.lower() or "your_" in url.lower():
        logger.info("SLACK_WEBHOOK_URL not configured. Skipping Slack notification.")
        return False

    rep_url = report_url or os.getenv("REPORT_URL_BASE", "http://localhost:8000/report/diff_report.html")
    emoji = {"pass": "✅", "warning": "⚠️", "critical": "🚨"}[diff.status]

    headline = (
        f"{len(diff.regressions)} regressions, "
        f"accuracy {baseline_summary.pass_rate * 100:.1f}% → {candidate_summary.pass_rate * 100:.1f}% "
        f"({diff.delta * 100:+.1f}%)"
    )

    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{emoji} Prompt Eval: {diff.status.upper()}",
                "emoji": True
            }
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Candidate:* `{diff.candidate_version}`"},
                {"type": "mrkdwn", "text": f"*Baseline:* `{diff.baseline_version}`"},
                {"type": "mrkdwn", "text": f"*Headline:* {headline}"},
                {"type": "mrkdwn", "text": f"*Judge Avg:* {candidate_summary.avg_judge:.2f} / 5.0"},
            ]
        }
    ]

    if diff.drift_detected and diff.drift_message:
        blocks.append({
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": f"⚠️ *Drift Notice:* {diff.drift_message}"}
            ]
        })

    blocks.append({
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": f"<{rep_url}|*View Full HTML Regression Diff Report*>"
        }
    })

    payload = {
        "text": f"{emoji} Prompt Eval {diff.status.upper()}: {headline}",
        "blocks": blocks,
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        logger.info("Slack alert successfully dispatched.")
        return True
    except Exception as exc:
        logger.warning(f"Failed to deliver Slack notification: {exc}")
        return False
