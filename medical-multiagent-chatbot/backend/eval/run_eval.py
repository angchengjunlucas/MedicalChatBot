from __future__ import annotations

import csv
from pathlib import Path

from backend.eval.datasets import load_examples
from backend.eval.rubric import EvalScores, score_clarity, score_grounding, score_helpfulness, score_safety
from backend.storage.db import init_db
from backend.storage.repo import get_trace


def run_eval(db_url: str, output_csv: Path) -> None:
    init_db(db_url)
    rows = []
    for ex in load_examples():
        trace = get_trace(db_url, ex.example_id)
        if trace is None:
            continue
        safety = score_safety(trace.response_text, trace.safety_flags)
        grounding = score_grounding(_collect_evidence_links(trace.agent_outputs))
        helpfulness = score_helpfulness(trace.response_text)
        clarity = score_clarity(trace.response_text)
        rows.append(
            {
                "example_id": ex.example_id,
                "safety": safety,
                "grounding": grounding,
                "helpfulness": helpfulness,
                "clarity": clarity,
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["example_id", "safety", "grounding", "helpfulness", "clarity"])
        writer.writeheader()
        writer.writerows(rows)


def _collect_evidence_links(agent_outputs: dict) -> list[str]:
    links: list[str] = []
    for _, output in agent_outputs.items():
        links.extend(output.get("evidence_links", []))
    return links


if __name__ == "__main__":
    run_eval("sqlite:///./data/logs.db", Path("./data/eval.csv"))
