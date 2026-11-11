from pathlib import Path
from src.models import CaseEvalResult, Category, RunSummary
from src.store import get_historical_pass_rates, get_latest_run_for_prompt, get_run, get_run_history, save_run


def test_sqlite_save_and_retrieve(tmp_path: Path):
    db_file = tmp_path / "test_runs.db"

    summary = RunSummary(
        run_id="run_v1_001",
        prompt_version="v1",
        model="gpt-4o-mini",
        dataset_version="1.0",
        created_at="2026-11-11T12:00:00Z",
        pass_rate=0.92,
        avg_judge=4.6,
        avg_latency_ms=150.0,
        total_tokens=1200,
        total_cases=10,
        passed_cases=9,
    )

    results = [
        CaseEvalResult(
            case_id="case_001",
            pred_category=Category.BILLING,
            pred_summary="Disputed charge",
            passed=True,
            judge_score=5,
            judge_reason="Accurate",
            latency_ms=120.0,
            tokens=45,
        ),
        CaseEvalResult(
            case_id="case_002",
            pred_category=Category.TECHNICAL,
            pred_summary="Crash report",
            passed=False,
            judge_score=2,
            judge_reason="Misclassified",
            latency_ms=180.0,
            tokens=50,
        )
    ]

    save_run(summary, results, db_path=db_file)

    retrieved = get_run("run_v1_001", db_path=db_file)
    assert retrieved is not None
    ret_summary, ret_results = retrieved

    assert ret_summary.prompt_version == "v1"
    assert ret_summary.pass_rate == 0.92
    assert len(ret_results) == 2
    assert ret_results[0].case_id == "case_001"
    assert ret_results[0].passed is True

    # Test get_latest_run_for_prompt
    latest = get_latest_run_for_prompt("v1", db_path=db_file)
    assert latest is not None
    assert latest[0].run_id == "run_v1_001"

    # Test pass rates history
    rates = get_historical_pass_rates(db_path=db_file)
    assert rates == [0.92]

    # Test history runs list
    history = get_run_history(limit=5, db_path=db_file)
    assert len(history) == 1
