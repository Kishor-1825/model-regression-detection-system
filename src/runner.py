from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timezone
from typing import List, Optional, Tuple
import uuid

from src.classifier import build_messages, _mock_classify
from src.judge import judge_summary_async
from src.models import (
    CaseEvalResult,
    ClassificationResult,
    GoldenCase,
    GoldenDataset,
    PromptConfig,
    RunSummary,
    is_valid_openai_key,
)

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None  # type: ignore


async def evaluate_single_case(
    case: GoldenCase,
    config: PromptConfig,
    aclient: Optional[AsyncOpenAI],
    semaphore: asyncio.Semaphore,
    mock: bool = False
) -> CaseEvalResult:
    """Evaluate one golden test case through the feature under test and judge."""
    async with semaphore:
        t0 = time.perf_counter()
        api_key = os.getenv("OPENAI_API_KEY")

        if mock or not is_valid_openai_key(api_key) or AsyncOpenAI is None or aclient is None:
            # Deterministic mock evaluation
            await asyncio.sleep(0.005)  # small async tick
            latency_ms = (time.perf_counter() - t0) * 1000
            pred = _mock_classify(case.email, config)
            passed = (pred.category == case.expected_category)

            judge = await judge_summary_async(
                email=case.email,
                expected=case.expected_summary,
                candidate=pred.summary,
                mock=True
            )

            return CaseEvalResult(
                case_id=case.id,
                pred_category=pred.category,
                pred_summary=pred.summary,
                passed=passed,
                judge_score=judge.score,
                judge_reason=judge.reason,
                latency_ms=round(latency_ms, 2),
                tokens=len(case.email.split()) * 2 + 35,
            )

        # Live OpenAI execution
        messages = build_messages(case.email, config)
        resp = await aclient.beta.chat.completions.parse(
            model=config.model,
            temperature=config.temperature,
            messages=messages,
            response_format=ClassificationResult,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        pred = resp.choices[0].message.parsed
        tokens = resp.usage.total_tokens if resp.usage else 0

        # Run LLM-as-judge on summary
        judge = await judge_summary_async(
            email=case.email,
            expected=case.expected_summary,
            candidate=pred.summary,
            aclient=aclient,
            mock=False
        )

        passed = (pred.category == case.expected_category)

        return CaseEvalResult(
            case_id=case.id,
            pred_category=pred.category,
            pred_summary=pred.summary,
            passed=passed,
            judge_score=judge.score,
            judge_reason=judge.reason,
            latency_ms=round(latency_ms, 2),
            tokens=tokens,
        )


async def run_evaluation(
    config: PromptConfig,
    dataset: GoldenDataset,
    concurrency: int = 8,
    mock: bool = False
) -> Tuple[RunSummary, List[CaseEvalResult]]:
    """Execute asynchronous evaluation across all cases in the dataset."""
    semaphore = asyncio.Semaphore(concurrency)
    api_key = os.getenv("OPENAI_API_KEY")

    aclient = None
    if not mock and is_valid_openai_key(api_key) and AsyncOpenAI is not None:
        aclient = AsyncOpenAI(api_key=api_key)

    tasks = [
        evaluate_single_case(
            case=c,
            config=config,
            aclient=aclient,
            semaphore=semaphore,
            mock=mock or (aclient is None)
        )
        for c in dataset.cases
    ]

    results: List[CaseEvalResult] = await asyncio.gather(*tasks)

    # Compute aggregate metrics
    total_cases = len(results)
    passed_cases = sum(1 for r in results if r.passed)
    pass_rate = round(passed_cases / max(1, total_cases), 4)
    avg_judge = round(sum(r.judge_score for r in results) / max(1, total_cases), 2)
    avg_latency = round(sum(r.latency_ms for r in results) / max(1, total_cases), 2)
    total_tokens = sum(r.tokens for r in results)

    run_id = f"run_{config.version}_{uuid.uuid4().hex[:8]}"
    created_at = datetime.now(timezone.utc).isoformat()

    summary = RunSummary(
        run_id=run_id,
        prompt_version=config.version,
        model=config.model,
        dataset_version=dataset.dataset_version,
        created_at=created_at,
        pass_rate=pass_rate,
        avg_judge=avg_judge,
        avg_latency_ms=avg_latency,
        total_tokens=total_tokens,
        total_cases=total_cases,
        passed_cases=passed_cases,
    )

    return summary, results
