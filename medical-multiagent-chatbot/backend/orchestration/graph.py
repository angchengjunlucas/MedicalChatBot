from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GraphNode:
    name: str
    description: str


class OrchestrationGraph:
    def __init__(self) -> None:
        self.nodes: list[GraphNode] = [
            GraphNode("triage", "Detect red flags / emergency routing."),
            GraphNode("rewrite", "Rewrite query for retrieval."),
            GraphNode("retrieve", "Hybrid retrieval from KB + PubMed."),
            GraphNode("route", "Supervisor selects specialists."),
            GraphNode("specialists", "Run specialist agents in parallel."),
            GraphNode("compose", "Compose final answer."),
            GraphNode("safety", "Safety postprocess."),
            GraphNode("log", "Persist trace and metrics."),
        ]

# lists stages for clarity of visualization