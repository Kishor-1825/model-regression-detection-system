from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
import yaml
from dotenv import load_dotenv

from src.diff import diff_runs
from src.models import ThresholdConfig, is_valid_openai_key, load_golden
from src.prompts import load_prompt
from src.report import generate_html_report, generate_pr_summary_markdown
from src.runner import run_evaluation
from src.slack import send_slack_alert
from src.store import (
    get_historical_pass_rates,
    get_latest_run_for_prompt,
    get_run_history,
    save_run,
)

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Model Regression Detection System (LLM CI/CD Eval Runner)"
    )
    parser.add_argument(
        "--candidate",
        type=str,
        default="v2",
        help="Candidate prompt version name (e.g. v2) or file path",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="v1",
        help="Baseline prompt version name (e.g. v1) or file path",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/golden_v1.json",
        help="Path to the golden dataset JSON file",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="thresholds.yaml",
        help="Path to thresholds.yaml configuration file",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="report",
        help="Output directory for generated diff reports",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="runs.db",
        help="SQLite database path for persisting run history",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Maximum concurrent asynchronous LLM evaluation requests",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force offline deterministic mock mode (skips live OpenAI API calls)",
    )
    return parser.parse_args()


def load_thresholds(config_path: str | Path = "thresholds.yaml") -> ThresholdConfig:
    p = Path(config_path)
    if not p.is_file():
        return ThresholdConfig()
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return ThresholdConfig.model_validate(data)


async def main_async(args: argparse.Namespace) -> int:
    print("\n=======================================================")
    print("  🚀 Model Regression Detection System (CI/CD Eval)    ")
    print("=======================================================\n")

    # 1. Load Configurations & Data
    cand_config = load_prompt(args.candidate)
    base_config = load_prompt(args.baseline)
    dataset = load_golden(args.dataset)
    thresholds = load_thresholds(args.thresholds)

    raw_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    has_valid_key = is_valid_openai_key(raw_api_key)

    if not args.mock and raw_api_key and not has_valid_key:
        print("[!] Notice: Detected placeholder or incomplete OPENAI_API_KEY in .env.")
        print("    -> Automatically falling back to OFFLINE MOCK mode (deterministic).")
        print("    -> To use live OpenAI calls, set a valid key in .env (https://platform.openai.com/api-keys).\n")

    is_mock = args.mock or not has_valid_key
    mode_str = "OFFLINE MOCK (Deterministic)" if is_mock else "LIVE OpenAI API (gpt-4o-mini)"
    print(f"[*] Execution Mode: {mode_str}")
    print(f"[*] Golden Dataset: {dataset.dataset_version} ({len(dataset.cases)} test cases)")
    print(f"[*] Candidate Prompt: '{cand_config.version}' | Baseline Prompt: '{base_config.version}'")
    print(f"[*] Thresholds: Warning drop >= {thresholds.warn_threshold:.1%}, Critical drop >= {thresholds.critical_threshold:.1%}\n")

    try:
        # 2. Run Baseline Evaluation (or check if already in SQLite)
        cached_baseline = get_latest_run_for_prompt(base_config.version, db_path=args.db)
        if cached_baseline and not args.mock:
            print(f"[*] Found cached baseline run in SQLite for '{base_config.version}'")
            base_summary, base_results = cached_baseline
        else:
            print(f"[*] Running evaluation for baseline '{base_config.version}'...")
            base_summary, base_results = await run_evaluation(
                config=base_config,
                dataset=dataset,
                concurrency=args.concurrency,
                mock=is_mock,
            )
            save_run(base_summary, base_results, db_path=args.db)
            print(f"    -> Baseline Pass Rate: {base_summary.pass_rate * 100:.1f}% (Judge Avg: {base_summary.avg_judge}/5.0)")

        # 3. Run Candidate Evaluation
        print(f"[*] Running evaluation for candidate '{cand_config.version}'...")
        cand_summary, cand_results = await run_evaluation(
            config=cand_config,
            dataset=dataset,
            concurrency=args.concurrency,
            mock=is_mock,
        )
        save_run(cand_summary, cand_results, db_path=args.db)
        print(f"    -> Candidate Pass Rate: {cand_summary.pass_rate * 100:.1f}% (Judge Avg: {cand_summary.avg_judge}/5.0)\n")
    except Exception as exc:
        err_msg = str(exc)
        if "AuthenticationError" in type(exc).__name__ or "Incorrect API key" in err_msg or "401" in err_msg:
            print(f"\n🚨 OpenAI Authentication Error: {err_msg}")
            print("   The OPENAI_API_KEY provided is rejected by OpenAI.")
            print("   Please check your API key at https://platform.openai.com/account/api-keys")
            print("   or run with --mock for offline deterministic evaluation.\n")
            return 1
        elif "RateLimitError" in type(exc).__name__ or "429" in err_msg:
            print(f"\n🚨 OpenAI Rate Limit / Quota Error: {err_msg}")
            print("   Please check your OpenAI quota or run with --mock for offline deterministic evaluation.\n")
            return 1
        elif "APIConnectionError" in type(exc).__name__:
            print(f"\n🚨 OpenAI Connection Error: {err_msg}")
            print("   Could not connect to OpenAI API. Check your network or run with --mock.\n")
            return 1
        raise

    # 4. Compute Diff & Drift
    history_pass_rates = get_historical_pass_rates(db_path=args.db)
    diff = diff_runs(
        baseline_results=base_results,
        candidate_results=cand_results,
        cases=dataset.cases,
        baseline_version=base_config.version,
        candidate_version=cand_config.version,
        thresholds=thresholds,
        history_pass_rates=history_pass_rates,
    )

    # 5. Generate Reports
    history_runs = get_run_history(limit=10, db_path=args.db)
    html_file = generate_html_report(
        diff=diff,
        baseline_summary=base_summary,
        candidate_summary=cand_summary,
        baseline_results=base_results,
        candidate_results=cand_results,
        cases=dataset.cases,
        history_runs=history_runs,
        out_dir=args.out,
    )
    summary_file = generate_pr_summary_markdown(
        diff=diff,
        baseline_summary=base_summary,
        candidate_summary=cand_summary,
        out_dir=args.out,
    )

    # 6. Send Slack Alert
    send_slack_alert(
        diff=diff,
        candidate_summary=cand_summary,
        baseline_summary=base_summary,
        report_url=str(html_file.resolve()),
    )

    # 7. Print Terminal Scorecard
    status_icon = {"pass": "✅", "warning": "⚠️", "critical": "🚨"}[diff.status]
    delta_str = f"+{diff.delta * 100:.1f}%" if diff.delta >= 0 else f"{diff.delta * 100:.1f}%"

    print("-------------------------------------------------------")
    print(f"  {status_icon} EVALUATION STATUS: {diff.status.upper()}")
    print("-------------------------------------------------------")
    print(f"  Baseline Accuracy  : {diff.baseline_pass_rate * 100:.1f}%")
    print(f"  Candidate Accuracy : {diff.candidate_pass_rate * 100:.1f}%")
    print(f"  Accuracy Delta     : {delta_str}")
    print(f"  Regressions        : {len(diff.regressions)} case(s) flipped to FAIL")
    print(f"  Improvements       : {len(diff.improvements)} case(s) flipped to PASS")
    print("  Per-Category Delta :")
    for cat, d in diff.per_category.items():
        print(f"    - {cat:<10}: {d * 100:+.1f}%")

    if diff.drift_detected:
        print(f"\n  [!] {diff.drift_message}")

    print(f"\n  [+] Full HTML Report : file:///{html_file.resolve().as_posix()}")
    print(f"  [+] PR Summary MD    : file:///{summary_file.resolve().as_posix()}")
    print("-------------------------------------------------------\n")

    if diff.status == "critical":
        print("🚨 CRITICAL REGRESSION: Exiting with code 1 to block PR merge!\n")
        return 1

    print("✅ EVALUATION PASSED: Safe to merge.\n")
    return 0


def main() -> None:
    code = asyncio.run(main_async(parse_args()))
    sys.exit(code)


if __name__ == "__main__":
    main()
