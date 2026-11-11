from __future__ import annotations

import os
from typing import Optional
from src.models import JudgeScore, is_valid_openai_key

try:
    from openai import OpenAI, AsyncOpenAI
except ImportError:
    OpenAI = None  # type: ignore
    AsyncOpenAI = None  # type: ignore

JUDGE_PROMPT = """You are an impartial, expert evaluation judge grading a one-sentence customer email summary.
Compare the Candidate Summary against the Reference Expected Summary for the given Customer Email.

Evaluation Rubric:
- 5 (Excellent): Preserves the exact core meaning, key entities, numbers, and intent without hallucinations or fluff.
- 4 (Good): Captures the main point accurately with minor missing nuance.
- 3 (Adequate): Partially correct or overly generic, but not misleading.
- 2 (Poor): Misses the central problem or misinterprets key facts.
- 1 (Unacceptable/Misleading): Completely wrong, hallucinates fabricated details, or contradicts the email.

Penalize invented details, hallucinated facts, and excessive wordiness.
Return a structured score (1-5) and a concise justification reason.
"""


def _mock_judge_summary(email: str, expected: str, candidate: str) -> JudgeScore:
    """Deterministic offline judge scorer for testing without live API keys."""
    if not candidate or len(candidate.strip()) < 5:
        return JudgeScore(score=1, reason="Candidate summary is empty or too short.")

    exp_words = set(expected.lower().split())
    cand_words = set(candidate.lower().split())
    overlap = len(exp_words.intersection(cand_words))
    ratio = overlap / max(1, len(exp_words))

    if ratio > 0.45:
        return JudgeScore(score=5, reason="Candidate accurately reflects reference meaning and key details.")
    elif ratio > 0.25:
        return JudgeScore(score=4, reason="Candidate captures primary intent with minor phrasing differences.")
    elif ratio > 0.10:
        return JudgeScore(score=3, reason="Candidate is acceptable but somewhat generic.")
    else:
        return JudgeScore(score=2, reason="Candidate summary lacks key reference concepts.")


def judge_summary(
    email: str,
    expected: str,
    candidate: str,
    client: Optional[OpenAI] = None,
    mock: bool = False
) -> JudgeScore:
    """Judge summary relevance synchronously."""
    api_key = os.getenv("OPENAI_API_KEY")
    if mock or not is_valid_openai_key(api_key) or OpenAI is None:
        return _mock_judge_summary(email, expected, candidate)

    if client is None:
        client = OpenAI(api_key=api_key)

    user_content = (
        f"Customer Email:\n{email}\n\n"
        f"Reference Expected Summary:\n{expected}\n\n"
        f"Candidate Summary:\n{candidate}"
    )

    resp = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        temperature=0.0,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format=JudgeScore,
    )
    return resp.choices[0].message.parsed


async def judge_summary_async(
    email: str,
    expected: str,
    candidate: str,
    aclient: Optional[AsyncOpenAI] = None,
    mock: bool = False
) -> JudgeScore:
    """Judge summary relevance asynchronously."""
    api_key = os.getenv("OPENAI_API_KEY")
    if mock or not is_valid_openai_key(api_key) or AsyncOpenAI is None:
        return _mock_judge_summary(email, expected, candidate)

    if aclient is None:
        aclient = AsyncOpenAI(api_key=api_key)

    user_content = (
        f"Customer Email:\n{email}\n\n"
        f"Reference Expected Summary:\n{expected}\n\n"
        f"Candidate Summary:\n{candidate}"
    )

    resp = await aclient.beta.chat.completions.parse(
        model="gpt-4o-mini",
        temperature=0.0,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format=JudgeScore,
    )
    return resp.choices[0].message.parsed
