from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyResult:
    text: str
    flags: list[str]


_DIAGNOSIS_REWRITES = [
    (r"\byou have\b", "One possible cause is", "diagnosis_language"),
    (r"\byou are diagnosed with\b", "One possible cause is", "diagnosis_language"),
    (r"\bthis is (likely|probably)\b", "This could be", "diagnosis_language"),
]

_MED_CHANGE_PATTERNS = [
    r"\bstop taking\b",
    r"\bstart taking\b",
    r"\bincrease\b",
    r"\bdecrease\b",
    r"\bchange (your|the) dose\b",
]

_DOSING_PATTERN = r"\b\d+(\.\d+)?\s?(mg|mcg|g|ml|iu|units)\b"


def apply_safety_rules(text: str) -> SafetyResult:
    flags: list[str] = []
    cleaned = text

    for pattern, replacement, flag in _DIAGNOSIS_REWRITES:
        if re.search(pattern, cleaned, flags=re.IGNORECASE):
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
            if flag not in flags:
                flags.append(flag)

    if _contains_medication_change(cleaned):
        cleaned = _mask_medication_phrases(cleaned)
        flags.append("medication_change_removed")

    if re.search(_DOSING_PATTERN, cleaned, flags=re.IGNORECASE):
        cleaned = re.sub(_DOSING_PATTERN, "[dosing removed]", cleaned, flags=re.IGNORECASE)
        flags.append("dosing_removed")

    return SafetyResult(text=cleaned.strip(), flags=flags)


def _contains_medication_change(text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in _MED_CHANGE_PATTERNS)


def _mask_medication_phrases(text: str) -> str:
    masked = text
    for pattern in _MED_CHANGE_PATTERNS:
        masked = re.sub(pattern, "[medication advice removed]", masked, flags=re.IGNORECASE)
    return masked
#this ensure that my chatbot doesnt say anything unsafe and overconfident
