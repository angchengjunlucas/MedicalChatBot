from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ChatExample:
    messages: list[dict[str, str]]


def load_chat_jsonl(path: str) -> list[ChatExample]:
    examples: list[ChatExample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            msgs = obj.get("messages")
            if not isinstance(msgs, list):
                raise ValueError("Each JSONL line must have a 'messages' list")
            examples.append(ChatExample(messages=msgs))
    return examples


def format_for_sft(messages: list[dict[str, str]]) -> str:
    parts = []
    for msg in messages:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts) + "\nASSISTANT:"

#this is code for loading and formatting training data for fine tuning