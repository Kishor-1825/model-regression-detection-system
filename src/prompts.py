from __future__ import annotations

from pathlib import Path
from typing import List
import yaml

from src.models import PromptConfig


def load_prompt(version_or_path: str, prompts_dir: str | Path = "prompts") -> PromptConfig:
    """Load a versioned prompt YAML file.
    Accepts either a version name (e.g. 'v1', 'v2') or a file path ('prompts/v1.yaml').
    """
    path = Path(version_or_path)
    if not path.is_file():
        # Check if version without extension inside prompts_dir
        candidate = Path(prompts_dir) / f"{version_or_path}.yaml"
        if candidate.is_file():
            path = candidate
        else:
            candidate_yml = Path(prompts_dir) / f"{version_or_path}.yml"
            if candidate_yml.is_file():
                path = candidate_yml
            else:
                raise FileNotFoundError(f"Prompt configuration not found for: '{version_or_path}' (searched {path} and {candidate})")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return PromptConfig.model_validate(data)


def list_prompts(prompts_dir: str | Path = "prompts") -> List[str]:
    """List all available prompt versions in prompts/ directory."""
    p_dir = Path(prompts_dir)
    if not p_dir.is_dir():
        return []
    return sorted([p.stem for p in p_dir.glob("*.yaml")] + [p.stem for p in p_dir.glob("*.yml")])
