import pytest
from pydantic import ValidationError

from src.models import (
    Category,
    ClassificationResult,
    GoldenCase,
    GoldenDataset,
    JudgeScore,
    PromptConfig,
    load_golden,
)


def test_category_enum():
    assert Category.BILLING == "billing"
    assert Category.TECHNICAL == "technical"
    assert Category.ACCOUNT == "account"
    assert Category.GENERAL == "general"


def test_classification_result_valid():
    res = ClassificationResult(
        category=Category.BILLING,
        summary="Customer disputes an invoice overcharge."
    )
    assert res.category == Category.BILLING
    assert "invoice overcharge" in res.summary


def test_classification_result_invalid():
    with pytest.raises(ValidationError):
        ClassificationResult(category="unknown_category", summary="Valid summary")


def test_judge_score_bounds():
    valid = JudgeScore(score=5, reason="Matches reference perfectly")
    assert valid.score == 5

    with pytest.raises(ValidationError):
        JudgeScore(score=6, reason="Too high")

    with pytest.raises(ValidationError):
        JudgeScore(score=0, reason="Too low")


def test_golden_dataset_loader():
    dataset = load_golden("data/golden_v1.json")
    assert dataset.dataset_version == "1.0"
    assert len(dataset.cases) >= 50

    first_case = dataset.cases[0]
    assert isinstance(first_case, GoldenCase)
    assert first_case.id == "case_001"
    assert first_case.expected_category == Category.BILLING
    assert first_case.difficulty in ["easy", "medium", "hard"]
