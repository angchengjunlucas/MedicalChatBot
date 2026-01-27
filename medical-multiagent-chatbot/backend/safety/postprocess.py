from __future__ import annotations

from backend.safety.rules import SafetyResult, apply_safety_rules


def safety_postprocess(text: str) -> SafetyResult:
    return apply_safety_rules(text)
