from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable


CARDIO_PATTERN = re.compile(
    r"\b(cardio|cardiac|cardiovascular|heart|myocard|c\.v\.s\.?|cvs)\b",
    re.IGNORECASE,
)
GERI_PATTERN = re.compile(
    r"\b("
    r"geriatr\w*|gerontology|elderly|older\s*adult|old\s*age|aging|ageing|senior|"
    r"frailty|falls?|dementia|alzheimer|delirium|sarcopenia|osteoporosis|"
    r"hip\s*fracture|polypharmacy|incontinence|pressure\s*ulcer|senile"
    r")\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Folder with MedMCQA json files")
    parser.add_argument("--output_dir", required=True, help="Output base folder")
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_examples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    files = [input_dir / "train.json", input_dir / "dev.json", input_dir / "test.json"]
    rows = list(load_rows(files))

    cardio_rows: list[dict[str, Any]] = []
    geri_rows: list[dict[str, Any]] = []
    overlap = 0
    dropped = 0

    for row in rows:
        is_cardio = matches_category(row, CARDIO_PATTERN, fields=("subject_name", "topic_name"))
        is_geri = matches_category(
            row,
            GERI_PATTERN,
            fields=("subject_name", "topic_name", "question", "exp"),
        )
        if is_cardio and is_geri:
            overlap += 1
            cardio_rows.append(row)
            continue
        if is_cardio:
            cardio_rows.append(row)
            continue
        if is_geri:
            geri_rows.append(row)
            continue
        dropped += 1

    report_counts(rows, cardio_rows, geri_rows, overlap, dropped)

    output_base = Path(args.output_dir)
    cardio_dir = output_base / "medmcqa_cardiology"
    geri_dir = output_base / "medmcqa_geriatrics"

    write_category("cardiology", cardio_rows, cardio_dir, args)
    write_category("geriatrics", geri_rows, geri_dir, args)


def load_rows(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        if not path.exists():
            raise SystemExit(f"Missing dataset file: {path}")
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                yield json.loads(line)


def matches_category(
    row: dict[str, Any],
    pattern: re.Pattern[str],
    fields: tuple[str, ...],
) -> bool:
    for field in fields:
        value = str(row.get(field) or "")
        if pattern.search(value):
            return True
    return False


def report_counts(
    all_rows: list[dict[str, Any]],
    cardio_rows: list[dict[str, Any]],
    geri_rows: list[dict[str, Any]],
    overlap: int,
    dropped: int,
) -> None:
    print(json.dumps(
        {
            "total_rows": len(all_rows),
            "cardiology_rows": len(cardio_rows),
            "geriatrics_rows": len(geri_rows),
            "overlap_rows_assigned_to_cardiology": overlap,
            "dropped_rows": dropped,
        },
        indent=2,
    ))


def write_category(
    name: str,
    rows: list[dict[str, Any]],
    out_dir: Path,
    args: argparse.Namespace,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    rows = rows[:]
    rng.shuffle(rows)

    max_examples = args.max_examples if args.max_examples and args.max_examples > 0 else None
    if max_examples is not None:
        rows = rows[:max_examples]

    test_count = int(round(len(rows) * args.test_size))
    test_rows = rows[:test_count]
    train_rows = rows[test_count:]

    train_path = out_dir / "train.jsonl"
    test_path = out_dir / "test.jsonl"

    train_stats = write_split(train_rows, train_path, name)
    test_stats = write_split(test_rows, test_path, name)

    meta = {
        "category": name,
        "train_path": str(train_path),
        "test_path": str(test_path),
        "train_written": train_stats["written"],
        "train_skipped": train_stats["skipped"],
        "test_written": test_stats["written"],
        "test_skipped": test_stats["skipped"],
        "seed": args.seed,
        "test_size": args.test_size,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))


def write_split(rows: list[dict[str, Any]], output_path: Path, category: str) -> dict[str, int]:
    written = 0
    skipped = 0
    with output_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows):
            record = build_record(row, idx, category)
            if record is None:
                skipped += 1
                continue
            f.write(json.dumps(record) + "\n")
            written += 1
    return {"written": written, "skipped": skipped}


def build_record(row: dict[str, Any], idx: int, category: str) -> dict[str, Any] | None:
    question = normalize_text(row.get("question"))
    if not question:
        return None

    options = [
        normalize_text(row.get("opa")),
        normalize_text(row.get("opb")),
        normalize_text(row.get("opc")),
        normalize_text(row.get("opd")),
    ]
    if all(not opt for opt in options):
        return None

    user_parts = [
        f"Category: {category}",
        f"Question: {question}",
        "Options:",
        f"A. {options[0]}",
        f"B. {options[1]}",
        f"C. {options[2]}",
        f"D. {options[3]}",
    ]
    user_content = "\n".join(user_parts)

    cop = row.get("cop")
    answer_letter = None
    if isinstance(cop, int) and 1 <= cop <= 4:
        answer_letter = ["A", "B", "C", "D"][cop - 1]

    exp = normalize_text(row.get("exp"))
    assistant_parts = []
    if answer_letter:
        answer_text = options[cop - 1] if 1 <= cop <= 4 else ""
        assistant_parts.append(f"Answer: {answer_letter}. {answer_text}".strip())
    if exp:
        assistant_parts.append(f"Explanation: {exp}")

    if not assistant_parts:
        return None

    messages = [
        {
            "role": "system",
            "content": (
                "You are a research-only medical assistant. Provide general educational "
                "information and do not diagnose or prescribe."
            ),
        },
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": "\n".join(assistant_parts)},
    ]

    example_id = row.get("id") or f"{category}:{idx}"
    return {"example_id": str(example_id), "messages": messages}


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


if __name__ == "__main__":
    main()
