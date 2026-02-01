from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalExample:
    example_id: str
    query: str


def load_examples() -> list[EvalExample]:
    return [
        EvalExample(example_id="ex1", query="I have chest pain after exercise"),
        EvalExample(example_id="ex2", query="Older adult with frequent falls and dizziness"),
        EvalExample(example_id="ex3", query="Feeling anxious with racing heart and insomnia"),
    ]
