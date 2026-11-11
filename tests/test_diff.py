import pytest
from src.diff import classify_status, detect_drift, diff_runs
from src.models import Category, GoldenCase, ThresholdConfig


def test_classify_status_thresholds():
    # Positive delta (improvement)
    assert classify_status(delta=0.05, warn=0.03, crit=0.08) == "pass"

    # Zero change
    assert classify_status(delta=0.0, warn=0.03, crit=0.08) == "pass"

    # Negligible drop (< 3%)
    assert classify_status(delta=-0.02, warn=0.03, crit=0.08) == "pass"

    # Warning drop (between 3% and 8%)
    assert classify_status(delta=-0.03, warn=0.03, crit=0.08) == "warning"
    assert classify_status(delta=-0.05, warn=0.03, crit=0.08) == "warning"
    assert classify_status(delta=-0.079, warn=0.03, crit=0.08) == "warning"

    # Critical drop (>= 8%)
    assert classify_status(delta=-0.08, warn=0.03, crit=0.08) == "critical"
    assert classify_status(delta=-0.15, warn=0.03, crit=0.08) == "critical"


def test_detect_drift():
    # Insufficient historical runs (< window)
    history_short = [0.95, 0.94, 0.95]
    drifted, msg = detect_drift(history_short, window=7, floor=0.90)
    assert not drifted
    assert msg is None

    # Healthy 7-run window (all >= 92%)
    history_healthy = [0.94, 0.93, 0.95, 0.92, 0.94, 0.93, 0.95]
    drifted, msg = detect_drift(history_healthy, window=7, floor=0.90)
    assert not drifted

    # Drifting 7-run window (average drops below 0.90)
    history_degraded = [0.92, 0.90, 0.89, 0.88, 0.87, 0.86, 0.88]
    drifted, msg = detect_drift(history_degraded, window=7, floor=0.90)
    assert drifted
    assert "Slow drift detected" in msg


def test_diff_runs_no_change():
    cases = [
        GoldenCase(id="c1", email="e1", expected_category=Category.BILLING, expected_summary="s1"),
        GoldenCase(id="c2", email="e2", expected_category=Category.TECHNICAL, expected_summary="s2"),
    ]

    base = [
        {"case_id": "c1", "passed": True, "pred_category": "billing", "pred_summary": "s1", "judge_score": 5, "latency_ms": 10, "tokens": 20},
        {"case_id": "c2", "passed": True, "pred_category": "technical", "pred_summary": "s2", "judge_score": 5, "latency_ms": 10, "tokens": 20},
    ]
    cand = [
        {"case_id": "c1", "passed": True, "pred_category": "billing", "pred_summary": "s1", "judge_score": 5, "latency_ms": 10, "tokens": 20},
        {"case_id": "c2", "passed": True, "pred_category": "technical", "pred_summary": "s2", "judge_score": 5, "latency_ms": 10, "tokens": 20},
    ]

    diff = diff_runs(base, cand, cases, "v1", "v2")
    assert diff.delta == 0.0
    assert diff.status == "pass"
    assert len(diff.regressions) == 0
    assert len(diff.improvements) == 0


def test_diff_runs_one_regression():
    cases = [
        GoldenCase(id="c1", email="e1", expected_category=Category.BILLING, expected_summary="s1"),
        GoldenCase(id="c2", email="e2", expected_category=Category.TECHNICAL, expected_summary="s2"),
    ]

    base = [
        {"case_id": "c1", "passed": True, "pred_category": "billing", "pred_summary": "s1", "judge_score": 5, "latency_ms": 10, "tokens": 20},
        {"case_id": "c2", "passed": True, "pred_category": "technical", "pred_summary": "s2", "judge_score": 5, "latency_ms": 10, "tokens": 20},
    ]
    cand = [
        {"case_id": "c1", "passed": True, "pred_category": "billing", "pred_summary": "s1", "judge_score": 5, "latency_ms": 10, "tokens": 20},
        {"case_id": "c2", "passed": False, "pred_category": "general", "pred_summary": "s2", "judge_score": 3, "latency_ms": 10, "tokens": 20},
    ]

    diff = diff_runs(base, cand, cases, "v1", "v2")
    assert diff.regressions == ["c2"]
    assert diff.improvements == []
    assert diff.delta == -0.5
    assert diff.status == "critical"  # -50% is >= 8% critical drop


def test_diff_runs_one_improvement():
    cases = [
        GoldenCase(id="c1", email="e1", expected_category=Category.BILLING, expected_summary="s1"),
        GoldenCase(id="c2", email="e2", expected_category=Category.TECHNICAL, expected_summary="s2"),
    ]

    base = [
        {"case_id": "c1", "passed": False, "pred_category": "general", "pred_summary": "s1", "judge_score": 2, "latency_ms": 10, "tokens": 20},
        {"case_id": "c2", "passed": True, "pred_category": "technical", "pred_summary": "s2", "judge_score": 5, "latency_ms": 10, "tokens": 20},
    ]
    cand = [
        {"case_id": "c1", "passed": True, "pred_category": "billing", "pred_summary": "s1", "judge_score": 5, "latency_ms": 10, "tokens": 20},
        {"case_id": "c2", "passed": True, "pred_category": "technical", "pred_summary": "s2", "judge_score": 5, "latency_ms": 10, "tokens": 20},
    ]

    diff = diff_runs(base, cand, cases, "v1", "v2")
    assert diff.regressions == []
    assert diff.improvements == ["c1"]
    assert diff.delta == +0.5
    assert diff.status == "pass"
    assert diff.per_category["billing"] == +1.0
