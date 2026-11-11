import pytest
from src.prompts import load_prompt, list_prompts


def test_list_prompts():
    prompts = list_prompts("prompts")
    assert "v1" in prompts
    assert "v2" in prompts
    assert "v3_regression" in prompts


def test_load_prompt_by_name():
    p1 = load_prompt("v1")
    assert p1.version == "v1"
    assert "triage assistant" in p1.system_prompt
    assert len(p1.few_shot_examples) > 0


def test_load_prompt_by_path():
    p2 = load_prompt("prompts/v2.yaml")
    assert p2.version == "v2"
    assert p2.temperature == 0.0


def test_load_prompt_nonexistent():
    with pytest.raises(FileNotFoundError):
        load_prompt("v999_does_not_exist")
