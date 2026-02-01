from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset name")
    parser.add_argument("--output_dir", required=True, help="Output folder for JSONL splits")
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_existing_splits", action="store_true")
    parser.add_argument("--messages_field", default="")
    parser.add_argument("--system_field", default="")
    parser.add_argument("--user_field", default="")
    parser.add_argument("--assistant_field", default="")
    parser.add_argument("--max_examples", type=int, default=0, help="Limit examples per split (0 = all)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset)
    train_ds, test_ds = resolve_splits(dataset, args)

    mapping = resolve_mapping(train_ds, args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.jsonl"
    test_path = out_dir / "test.jsonl"

    train_stats = write_split(train_ds, train_path, mapping, args)
    test_stats = write_split(test_ds, test_path, mapping, args)

    meta = {
        "dataset": args.dataset,
        "train_path": str(train_path),
        "test_path": str(test_path),
        "train_written": train_stats["written"],
        "train_skipped": train_stats["skipped"],
        "test_written": test_stats["written"],
        "test_skipped": test_stats["skipped"],
        "mapping": mapping,
        "seed": args.seed,
        "test_size": args.test_size,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))


def resolve_splits(dataset: DatasetDict, args: argparse.Namespace) -> tuple[Dataset, Dataset]:
    if not isinstance(dataset, DatasetDict):
        raise SystemExit("Expected a DatasetDict from load_dataset")

    if args.use_existing_splits and "train" in dataset and ("test" in dataset or "validation" in dataset):
        train = dataset["train"]
        test = dataset["test"] if "test" in dataset else dataset["validation"]
        return train, test

    if "train" in dataset and len(dataset.keys()) == 1:
        base = dataset["train"]
    elif "train" in dataset:
        base = dataset["train"]
    else:
        base = concatenate_datasets([dataset[split] for split in dataset.keys()])

    split = base.train_test_split(test_size=args.test_size, seed=args.seed, shuffle=True)
    return split["train"], split["test"]


def resolve_mapping(dataset: Dataset, args: argparse.Namespace) -> dict[str, str]:
    if args.messages_field:
        return {"messages_field": args.messages_field}

    if args.user_field and args.assistant_field:
        mapping = {
            "user_field": args.user_field,
            "assistant_field": args.assistant_field,
        }
        if args.system_field:
            mapping["system_field"] = args.system_field
        return mapping

    columns = set(dataset.column_names)
    if "messages" in columns:
        return {"messages_field": "messages"}
    if {"instruction", "input", "output"} <= columns:
        return {
            "system_field": "instruction",
            "user_field": "input",
            "assistant_field": "output",
        }
    if {"prompt", "response"} <= columns:
        return {"user_field": "prompt", "assistant_field": "response"}
    if {"question", "answer"} <= columns:
        return {"user_field": "question", "assistant_field": "answer"}
    if {"query", "response"} <= columns:
        return {"user_field": "query", "assistant_field": "response"}

    raise SystemExit(
        "Unable to infer field mapping. Provide --messages_field or "
        "--system_field/--user_field/--assistant_field."
    )


def write_split(
    dataset: Dataset,
    output_path: Path,
    mapping: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, int]:
    written = 0
    skipped = 0
    limit = args.max_examples if args.max_examples and args.max_examples > 0 else None

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, example in enumerate(dataset):
            if limit is not None and written >= limit:
                break
            messages = build_messages(example, mapping)
            if not messages:
                skipped += 1
                continue
            record = {
                "example_id": f"{args.dataset}:{idx}",
                "messages": messages,
            }
            f.write(json.dumps(record) + "\n")
            written += 1

    return {"written": written, "skipped": skipped}


def build_messages(example: dict[str, Any], mapping: dict[str, str]) -> list[dict[str, str]] | None:
    if "messages_field" in mapping:
        return normalize_messages(example.get(mapping["messages_field"]))

    system_field = mapping.get("system_field")
    user_field = mapping["user_field"]
    assistant_field = mapping["assistant_field"]

    messages: list[dict[str, str]] = []
    if system_field:
        system_text = normalize_text(example.get(system_field, ""))
        if system_text:
            messages.append({"role": "system", "content": system_text})

    user_text = normalize_text(example.get(user_field, ""))
    assistant_text = normalize_text(example.get(assistant_field, ""))
    if not user_text or not assistant_text:
        return None

    messages.append({"role": "user", "content": user_text})
    messages.append({"role": "assistant", "content": assistant_text})
    return messages


def normalize_messages(raw: Any) -> list[dict[str, str]] | None:
    if raw is None:
        return None
    if not isinstance(raw, list):
        return None

    normalized: list[dict[str, str]] = []
    for msg in raw:
        if isinstance(msg, dict):
            role = msg.get("role") or msg.get("from")
            content = msg.get("content") or msg.get("value") or msg.get("text")
        elif isinstance(msg, (list, tuple)) and len(msg) == 2:
            role, content = msg
        else:
            continue

        role = normalize_role(str(role)) if role else "user"
        content = normalize_text(content)
        if not content:
            continue
        normalized.append({"role": role, "content": content})

    if not normalized:
        return None
    return normalized


def normalize_role(role: str) -> str:
    role = role.strip().lower()
    if role in {"human", "user"}:
        return "user"
    if role in {"assistant", "gpt", "bot"}:
        return "assistant"
    if role == "system":
        return "system"
    return "user"


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


if __name__ == "__main__":
    main()
