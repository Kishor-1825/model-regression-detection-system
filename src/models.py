from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class Category(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    GENERAL = "general"


class ClassificationResult(BaseModel):
    category: Category = Field(description="The primary category of the customer email")
    summary: str = Field(description="A concise, factual one-sentence summary of the customer's intent or problem")


class FewShotExample(BaseModel):
    email: str
    category: Category
    summary: str


class PromptConfig(BaseModel):
    version: str
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    created_at: Optional[str] = None
    description: Optional[str] = None
    system_prompt: str
    few_shot_examples: List[FewShotExample] = Field(default_factory=list)


class GoldenCase(BaseModel):
    id: str
    email: str
    expected_category: Category
    expected_summary: str
    difficulty: Literal["easy", "medium", "hard"] = "easy"
    notes: Optional[str] = ""


class GoldenDataset(BaseModel):
    dataset_version: str
    description: Optional[str] = ""
    cases: List[GoldenCase]


def load_golden(path: str | Path = "data/golden_v1.json") -> GoldenDataset:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Golden dataset not found at: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return GoldenDataset.model_validate(data)


class JudgeScore(BaseModel):
    score: int = Field(ge=1, le=5, description="1 (poor/misleading) to 5 (accurate & comprehensive)")
    reason: str = Field(description="Clear explanation of the evaluation grade")


class CaseEvalResult(BaseModel):
    case_id: str
    pred_category: Category
    pred_summary: str
    passed: bool
    judge_score: int
    judge_reason: str = ""
    latency_ms: float = 0.0
    tokens: int = 0


class RunSummary(BaseModel):
    run_id: str
    prompt_version: str
    model: str
    dataset_version: str
    created_at: str
    pass_rate: float
    avg_judge: float
    avg_latency_ms: float
    total_tokens: int
    total_cases: int
    passed_cases: int


class DiffResult(BaseModel):
    baseline_version: str
    candidate_version: str
    baseline_pass_rate: float
    candidate_pass_rate: float
    delta: float
    status: Literal["pass", "warning", "critical"]
    per_category: Dict[str, float]
    regressions: List[str]
    improvements: List[str]
    drift_detected: bool = False
    drift_message: Optional[str] = None


class ThresholdConfig(BaseModel):
    warn_threshold: float = 0.03
    critical_threshold: float = 0.08
    drift_window: int = 7
    drift_floor: float = 0.90
    max_concurrency: int = 8
    timeout_seconds: int = 30


def is_valid_openai_key(key: Optional[str]) -> bool:
    """Return True if key looks like a real, non-placeholder OpenAI API key."""
    if not key or not isinstance(key, str):
        return False
    k = key.strip()
    if not k:
        return False
    # Check common dummy/placeholder patterns
    if (
        k.startswith("your_")
        or "xxxx" in k.lower()
        or "placeholder" in k.lower()
        or "dummy" in k.lower()
        or "actualsecretkey" in k.lower()
        or "yourrealkey" in k.lower()
        or "..." in k
        or "<" in k
        or ">" in k
    ):
        return False
    # Valid OpenAI keys start with sk- and have reasonable length (typically >= 20 characters)
    if not k.startswith("sk-") or len(k) < 20:
        return False
    return True

