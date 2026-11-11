import pytest
from pathlib import Path
from src.diff import diff_runs
from src.models import load_golden
from src.prompts import load_prompt
from src.report import generate_html_report, generate_pr_summary_markdown
from src.runner import run_evaluation


@pytest.mark.asyncio
async def test_run_evaluation_mock():
    config = load_prompt("v1")
    dataset = load_golden("data/golden_v1.json")

    # Run on a subset of test cases for fast test execution
    sub_dataset = dataset.model_copy(update={"cases": dataset.cases[:8]})

    summary, results = await run_evaluation(
        config=config,
        dataset=sub_dataset,
        concurrency=4,
        mock=True
    )

    assert summary.prompt_version == "v1"
    assert summary.total_cases == 8
    assert len(results) == 8
    assert summary.pass_rate > 0.0
    assert summary.avg_judge >= 1.0


@pytest.mark.asyncio
async def test_diff_and_reporting_end_to_end(tmp_path: Path):
    v1_config = load_prompt("v1")
    v2_config = load_prompt("v2")
    dataset = load_golden("data/golden_v1.json")

    sub_dataset = dataset.model_copy(update={"cases": dataset.cases[:10]})

    s1, r1 = await run_evaluation(v1_config, sub_dataset, concurrency=4, mock=True)
    s2, r2 = await run_evaluation(v2_config, sub_dataset, concurrency=4, mock=True)

    diff = diff_runs(r1, r2, sub_dataset.cases, "v1", "v2")
    assert diff.status in ["pass", "warning", "critical"]

    html_file = generate_html_report(
        diff=diff,
        baseline_summary=s1,
        candidate_summary=s2,
        baseline_results=r1,
        candidate_results=r2,
        cases=sub_dataset.cases,
        out_dir=tmp_path / "report"
    )
    assert html_file.is_file()
    assert "LLM Regression Diff Report" in html_file.read_text(encoding="utf-8")

    summary_md = generate_pr_summary_markdown(
        diff=diff,
        baseline_summary=s1,
        candidate_summary=s2,
        out_dir=tmp_path / "report"
    )
    assert summary_md.is_file()
    assert "Model Regression Eval" in summary_md.read_text(encoding="utf-8")
