from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from src.models import CaseEvalResult, DiffResult, GoldenCase, ThresholdConfig


def classify_status(
    delta: float,
    warn: float = 0.03,
    crit: float = 0.08
) -> Literal["pass", "warning", "critical"]:
    """Classify the health status based on pass rate delta.
    Delta is (candidate_pass_rate - baseline_pass_rate).
    A drop is negative delta.
    """
    drop = -delta
    if drop >= crit:
        return "critical"
    if drop >= warn:
        return "warning"
    return "pass"


def detect_drift(
    history_pass_rates: List[float],
    window: int = 7,
    floor: float = 0.90
) -> Tuple[bool, Optional[str]]:
    """Detect slow drift over a rolling window of historical evaluations.
    Catches gradual degradation where no single run triggered an alert,
    but moving average drops below acceptable standards.
    """
    if len(history_pass_rates) < window:
        return False, None

    recent_slice = history_pass_rates[-window:]
    rolling_avg = sum(recent_slice) / window
    if rolling_avg < floor:
        msg = (
            f"Slow drift detected: {window}-run rolling average ({rolling_avg:.1%}) "
            f"dropped below quality floor ({floor:.1%})."
        )
        return True, msg

    return False, None


def _to_dict_map(
    results: Union[List[CaseEvalResult], List[Dict[str, Any]]]
) -> Dict[str, Dict[str, Any]]:
    """Normalize CaseEvalResult objects or dicts into a dictionary keyed by case_id."""
    res_map: Dict[str, Dict[str, Any]] = {}
    for r in results:
        if isinstance(r, CaseEvalResult):
            res_map[r.case_id] = {
                "case_id": r.case_id,
                "passed": r.passed,
                "pred_category": r.pred_category.value,
                "pred_summary": r.pred_summary,
                "judge_score": r.judge_score,
                "latency_ms": r.latency_ms,
                "tokens": r.tokens,
            }
        elif isinstance(r, dict):
            res_map[r["case_id"]] = r
        else:
            raise TypeError(f"Unsupported result type: {type(r)}")
    return res_map


def diff_runs(
    baseline_results: Union[List[CaseEvalResult], List[Dict[str, Any]]],
    candidate_results: Union[List[CaseEvalResult], List[Dict[str, Any]]],
    cases: List[GoldenCase],
    baseline_version: str = "baseline",
    candidate_version: str = "candidate",
    thresholds: Optional[ThresholdConfig] = None,
    history_pass_rates: Optional[List[float]] = None,
) -> DiffResult:
    """Compare candidate evaluation against baseline.
    Computes overall pass rate delta, per-category delta, regressions, improvements,
    status classification, and rolling drift.
    """
    if thresholds is None:
        thresholds = ThresholdConfig()

    b_map = _to_dict_map(baseline_results)
    n_map = _to_dict_map(candidate_results)

    # Regressions: passed in baseline, failed in candidate
    regressions = [
        cid for cid in n_map
        if cid in b_map and b_map[cid]["passed"] and not n_map[cid]["passed"]
    ]

    # Improvements: failed in baseline, passed in candidate
    improvements = [
        cid for cid in n_map
        if cid in b_map and not b_map[cid]["passed"] and n_map[cid]["passed"]
    ]

    def calc_rate(items: List[Dict[str, Any]]) -> float:
        if not items:
            return 0.0
        return sum(1 for item in items if item["passed"]) / len(items)

    b_pass_rate = calc_rate(list(b_map.values()))
    c_pass_rate = calc_rate(list(n_map.values()))
    delta = c_pass_rate - b_pass_rate

    # Per-category accuracy delta
    per_category: Dict[str, float] = {}
    distinct_categories = {c.expected_category.value for c in cases}
    for cat in sorted(distinct_categories):
        cat_case_ids = [c.id for c in cases if c.expected_category.value == cat]
        b_cat_items = [b_map[cid] for cid in cat_case_ids if cid in b_map]
        c_cat_items = [n_map[cid] for cid in cat_case_ids if cid in n_map]

        b_rate = calc_rate(b_cat_items)
        c_rate = calc_rate(c_cat_items)
        per_category[cat] = round(c_rate - b_rate, 4)

    status = classify_status(
        delta=delta,
        warn=thresholds.warn_threshold,
        crit=thresholds.critical_threshold
    )

    # Drift check
    drift_detected = False
    drift_message = None
    if history_pass_rates is not None:
        drift_detected, drift_message = detect_drift(
            history_pass_rates=history_pass_rates,
            window=thresholds.drift_window,
            floor=thresholds.drift_floor,
        )

    return DiffResult(
        baseline_version=baseline_version,
        candidate_version=candidate_version,
        baseline_pass_rate=round(b_pass_rate, 4),
        candidate_pass_rate=round(c_pass_rate, 4),
        delta=round(delta, 4),
        status=status,
        per_category=per_category,
        regressions=regressions,
        improvements=improvements,
        drift_detected=drift_detected,
        drift_message=drift_message,
    )
