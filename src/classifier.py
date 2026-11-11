from __future__ import annotations

import os
import json
import re
from typing import Any, Dict, List, Optional

from src.models import Category, ClassificationResult, PromptConfig, is_valid_openai_key

try:
    from openai import OpenAI, AsyncOpenAI
except ImportError:
    OpenAI = None  # type: ignore
    AsyncOpenAI = None  # type: ignore


def build_messages(email: str, config: PromptConfig) -> List[Dict[str, str]]:
    """Build standard OpenAI chat messages array from PromptConfig and email input."""
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": config.system_prompt}
    ]

    for ex in config.few_shot_examples:
        messages.append({"role": "user", "content": f"Customer Email:\n{ex.email}"})
        assistant_content = json.dumps({
            "category": ex.category.value,
            "summary": ex.summary
        })
        messages.append({"role": "assistant", "content": assistant_content})

    messages.append({"role": "user", "content": f"Customer Email:\n{email}"})
    return messages


def _mock_classify(email: str, config: PromptConfig) -> ClassificationResult:
    """Deterministic mock classifier used for offline test suites, CI checks without API keys,

    or demonstration runs. Emulates prompt quality differences:
    - v1: High baseline accuracy with difficulty handling sarcasm / edge cases.
    - v2: Superior performance on edge cases & sarcasm.
    - v3_regression: Degraded behavior that triggers critical regressions.
    """
    lower = email.lower()
    version = config.version.lower()

    # If intentionally degraded prompt v3_regression:
    if "v3" in version or "regression" in version:
        # Fails on several billing and technical cases, incorrectly categorizing as general
        if "outage" in lower or "surcharge" in lower or "charge" in lower or "refund" in lower:
            return ClassificationResult(
                category=Category.GENERAL,
                summary="Customer had feedback regarding service or general charges."
            )
        if "password" in lower or "login" in lower:
            return ClassificationResult(
                category=Category.ACCOUNT,
                summary="Customer needs account assistance."
            )
        return ClassificationResult(
            category=Category.GENERAL,
            summary="General customer inquiry."
        )

    # Candidate v2 handles sarcasm and tricky edge cases accurately
    if "v2" in version:
        if "outage" in lower or "another outage" in lower:
            return ClassificationResult(
                category=Category.TECHNICAL,
                summary="Customer sarcastically complains about repeated service downtime."
            )
        if "unable to login because my billing card failed" in lower:
            return ClassificationResult(
                category=Category.BILLING,
                summary="Customer cannot login due to a payment card failure."
            )

    # Default baseline v1 heuristics
    # Edge case sarcasm in v1 might misclassify as general
    if "another outage" in lower and "v1" in version:
        return ClassificationResult(
            category=Category.GENERAL,
            summary="Customer expresses frustration with subscription."
        )

    if any(k in lower for k in ["bill", "charged", "invoice", "refund", "downgrade", "upgrade", "surcharge", "discount", "pricing", "w-9", "vat", "auto-debit", "facture"]):
        return ClassificationResult(
            category=Category.BILLING,
            summary=f"Customer has a billing or invoice inquiry regarding {re.sub(r'[^a-zA-Z0-9 ]', '', email[:40]).strip()}."
        )
    elif any(k in lower for k in ["outage", "error", "bug", "crash", "timeout", "500", "latency", "sdk", "websocket", "indexing", "saml", "unicode", "broken", "429"]):
        return ClassificationResult(
            category=Category.TECHNICAL,
            summary=f"Customer reports a technical error or bug: {email[:45].strip()}..."
        )
    elif any(k in lower for k in ["password", "2fa", "login", "sso", "authenticator", "gdpr", "permission", "contraseña", "locked"]):
        return ClassificationResult(
            category=Category.ACCOUNT,
            summary="Customer requires assistance with account authentication, credentials, or security settings."
        )
    else:
        return ClassificationResult(
            category=Category.GENERAL,
            summary="Customer is submitting a general question or company inquiry."
        )


def classify_email(
    email: str,
    config: PromptConfig,
    client: Optional[OpenAI] = None,
    mock: bool = False
) -> ClassificationResult:
    """Synchronous email classification feature under test."""
    api_key = os.getenv("OPENAI_API_KEY")
    if mock or not is_valid_openai_key(api_key) or OpenAI is None:
        return _mock_classify(email, config)

    if client is None:
        client = OpenAI(api_key=api_key)

    messages = build_messages(email, config)
    resp = client.beta.chat.completions.parse(
        model=config.model,
        temperature=config.temperature,
        messages=messages,
        response_format=ClassificationResult,
    )
    return resp.choices[0].message.parsed
