from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List, Optional, Tuple

from src.models import CaseEvalResult, Category, RunSummary


def get_connection(db_path: str | Path = "runs.db") -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | Path = "runs.db") -> None:
    """Initialize SQLite tables for runs and results."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                prompt_version TEXT NOT NULL,
                model TEXT NOT NULL,
                dataset_version TEXT NOT NULL,
                created_at TEXT NOT NULL,
                pass_rate REAL NOT NULL,
                avg_judge REAL NOT NULL,
                avg_latency_ms REAL NOT NULL,
                total_tokens INTEGER NOT NULL,
                total_cases INTEGER NOT NULL,
                passed_cases INTEGER NOT NULL
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS results (
                run_id TEXT NOT NULL,
                case_id TEXT NOT NULL,
                pred_category TEXT NOT NULL,
                pred_summary TEXT NOT NULL,
                passed INTEGER NOT NULL,
                judge_score INTEGER NOT NULL,
                judge_reason TEXT,
                latency_ms REAL NOT NULL,
                tokens INTEGER NOT NULL,
                PRIMARY KEY (run_id, case_id),
                FOREIGN KEY (run_id) REFERENCES runs (run_id) ON DELETE CASCADE
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_prompt ON runs(prompt_version);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_results_run_id ON results(run_id);")
        conn.commit()


def save_run(
    summary: RunSummary,
    results: List[CaseEvalResult],
    db_path: str | Path = "runs.db"
) -> None:
    """Persist a complete evaluation run summary and its individual case results."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO runs (
                run_id, prompt_version, model, dataset_version, created_at,
                pass_rate, avg_judge, avg_latency_ms, total_tokens, total_cases, passed_cases
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            summary.run_id, summary.prompt_version, summary.model, summary.dataset_version,
            summary.created_at, summary.pass_rate, summary.avg_judge, summary.avg_latency_ms,
            summary.total_tokens, summary.total_cases, summary.passed_cases
        ))

        cursor.executemany("""
            INSERT OR REPLACE INTO results (
                run_id, case_id, pred_category, pred_summary, passed,
                judge_score, judge_reason, latency_ms, tokens
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            (
                summary.run_id, r.case_id, r.pred_category.value, r.pred_summary,
                1 if r.passed else 0, r.judge_score, r.judge_reason, r.latency_ms, r.tokens
            )
            for r in results
        ])
        conn.commit()


def get_run(
    run_id: str,
    db_path: str | Path = "runs.db"
) -> Optional[Tuple[RunSummary, List[CaseEvalResult]]]:
    """Retrieve run summary and its case results by run_id."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        if not row:
            return None

        summary = RunSummary(
            run_id=row["run_id"],
            prompt_version=row["prompt_version"],
            model=row["model"],
            dataset_version=row["dataset_version"],
            created_at=row["created_at"],
            pass_rate=row["pass_rate"],
            avg_judge=row["avg_judge"],
            avg_latency_ms=row["avg_latency_ms"],
            total_tokens=row["total_tokens"],
            total_cases=row["total_cases"],
            passed_cases=row["passed_cases"],
        )

        cursor.execute("SELECT * FROM results WHERE run_id = ?", (run_id,))
        results = [
            CaseEvalResult(
                case_id=r["case_id"],
                pred_category=Category(r["pred_category"]),
                pred_summary=r["pred_summary"],
                passed=bool(r["passed"]),
                judge_score=r["judge_score"],
                judge_reason=r["judge_reason"] or "",
                latency_ms=r["latency_ms"],
                tokens=r["tokens"],
            )
            for r in cursor.fetchall()
        ]
        return summary, results


def get_latest_run_for_prompt(
    prompt_version: str,
    db_path: str | Path = "runs.db"
) -> Optional[Tuple[RunSummary, List[CaseEvalResult]]]:
    """Get the most recent run for a specific prompt version."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT run_id FROM runs WHERE prompt_version = ? ORDER BY created_at DESC LIMIT 1",
            (prompt_version,)
        )
        row = cursor.fetchone()
        if not row:
            return None
        return get_run(row["run_id"], db_path=db_path)


def get_run_history(
    limit: int = 15,
    db_path: str | Path = "runs.db"
) -> List[RunSummary]:
    """Retrieve the recent run history ordered chronologically."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM runs ORDER BY created_at ASC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        return [
            RunSummary(
                run_id=r["run_id"],
                prompt_version=r["prompt_version"],
                model=r["model"],
                dataset_version=r["dataset_version"],
                created_at=r["created_at"],
                pass_rate=r["pass_rate"],
                avg_judge=r["avg_judge"],
                avg_latency_ms=r["avg_latency_ms"],
                total_tokens=r["total_tokens"],
                total_cases=r["total_cases"],
                passed_cases=r["passed_cases"],
            )
            for r in rows
        ]


def get_historical_pass_rates(db_path: str | Path = "runs.db") -> List[float]:
    """Get all past pass rates in chronological order for rolling drift detection."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT pass_rate FROM runs ORDER BY created_at ASC")
        return [row["pass_rate"] for row in cursor.fetchall()]
