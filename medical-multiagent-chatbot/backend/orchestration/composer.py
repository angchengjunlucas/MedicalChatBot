from __future__ import annotations

from dataclasses import dataclass

from backend.agents.base import AgentOutput


@dataclass(frozen=True)
class ComposedAnswer:
    short_text: str
    full_text: str


class FinalComposer:
    def compose(self, query: str, outputs: dict[str, AgentOutput], rationale: str) -> ComposedAnswer:
        if not outputs:
            fallback = (
                "This is general information, not medical advice. Only a clinician can diagnose or prescribe.\n\n"
                "No specialist output was available for your question. Please consider seeking medical advice."
            )
            return ComposedAnswer(short_text=fallback, full_text=fallback)

        summaries = [o.summary for o in outputs.values() if o.summary]
        key_points = _dedupe([p for o in outputs.values() for p in o.key_points])
        red_flags = _dedupe([p for o in outputs.values() for p in o.red_flags])
        uncertainties = _dedupe([p for o in outputs.values() for p in o.uncertainties])
        evidence = _dedupe([e for o in outputs.values() for e in o.evidence_links])
        drafts = [o.draft_response_text for o in outputs.values() if o.draft_response_text]

        no_evidence = not evidence or all(e == "NO_EVIDENCE" for e in evidence)

        short_parts = [
            "This is general information, not medical advice. Only a clinician can diagnose or prescribe.",
            "",
        ]

        if no_evidence:
            short_parts.append("No evidence was retrieved; the following is general information only.")
            short_parts.append("")

        draft = _select_draft(drafts)
        if draft:
            formatted_draft = _format_draft(draft)
            short_parts.extend(_split_lines(formatted_draft))
            has_warning_section = "when to seek urgent care" in formatted_draft.lower()
            if not has_warning_section:
                warnings = _dedupe(red_flags)
                cleaned_red = [p for p in (_strip_citations(p) for p in warnings) if p]
                if cleaned_red:
                    short_parts.append("")
                    short_parts.append("Urgent warning signs (seek care if any apply):")
                    short_parts.extend([f"- {p}" for p in cleaned_red[:4]])
        else:
            causes, checks, warnings = _parse_labeled_summary(summaries[0] if summaries else "")
            if causes:
                short_parts.append("Possible causes (not a diagnosis):")
                short_parts.extend([f"- {item}" for item in causes[:4]])
                short_parts.append("")
            if checks:
                short_parts.append("What to check first (with a clinician):")
                short_parts.extend([f"- {item}" for item in checks[:4]])
                short_parts.append("")

            if not causes and not checks:
                cleaned_points = [p for p in (_strip_citations(k) for k in key_points) if p]
                if cleaned_points:
                    short_parts.append("Top takeaways:")
                    short_parts.extend([f"- {p}" for p in cleaned_points[:4]])
                    short_parts.append("")
                elif summaries:
                    short_parts.append("Top takeaway:")
                    short_parts.append(f"- {summaries[0]}")
                    short_parts.append("")

            if warnings:
                short_parts.append("Urgent warning signs (seek care if any apply):")
                short_parts.extend([f"- {item}" for item in warnings[:4]])
                short_parts.append("")
            elif red_flags:
                cleaned_red = [p for p in (_strip_citations(p) for p in red_flags) if p]
                if cleaned_red:
                    short_parts.append("Urgent warning signs (seek care if any apply):")
                    short_parts.extend([f"- {p}" for p in cleaned_red[:4]])
                    short_parts.append("")

        full_parts = []
        full_parts.append("This is general information, not medical advice. Only a clinician can diagnose or prescribe.")
        full_parts.append("")
        full_parts.append(f"Question: {query}")
        full_parts.append("")
        if summaries:
            full_parts.append("Summary:")
            if no_evidence:
                full_parts.append("No evidence was retrieved; the following is general information only.")
            full_parts.append(" ".join(summaries))
            full_parts.append("")
        if key_points:
            full_parts.append("Key points:")
            full_parts.extend([f"- {p}" for p in key_points])
            full_parts.append("")
        if red_flags:
            full_parts.append("Red flags (seek urgent care if any apply):")
            full_parts.extend([f"- {p}" for p in red_flags])
            full_parts.append("")
        if uncertainties:
            full_parts.append("Uncertainties / limitations:")
            full_parts.extend([f"- {p}" for p in uncertainties])
            full_parts.append("")
        full_parts.append("Questions to ask a clinician:")
        full_parts.extend(
            [
                "- What are the most likely causes given my symptoms and history?",
                "- Are any tests needed right now?",
                "- What warning signs should prompt urgent care?",
            ]
        )
        full_parts.append("")
        if evidence:
            full_parts.append("Sources (evidence IDs):")
            full_parts.extend([f"- {e}" for e in evidence])
            full_parts.append("")
        if rationale:
            full_parts.append(f"Supervisor rationale: {rationale}")

        return ComposedAnswer(
            short_text="\n".join(short_parts).strip(),
            full_text="\n".join(full_parts).strip(),
        )


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for item in items:
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _strip_citations(text: str) -> str:
    import re

    cleaned = re.sub(r"\s*\\((?:PMID|KB):[^)]*\\)", "", text)
    return cleaned.strip()


def _parse_labeled_summary(text: str) -> tuple[list[str], list[str], list[str]]:
    if not text:
        return [], [], []
    causes = _split_items(_extract_label(text, "Likely causes"))
    checks = _split_items(_extract_label(text, "First checks"))
    warnings = _split_items(_extract_label(text, "Urgent red flags"))
    return causes, checks, warnings


def _extract_label(text: str, label: str) -> str:
    lower = text.lower()
    target = label.lower()
    start = lower.find(target)
    if start == -1:
        return ""
    colon = text.find(":", start)
    if colon == -1:
        return ""
    content_start = colon + 1
    next_labels = ["likely causes", "first checks", "urgent red flags"]
    next_positions = []
    for lab in next_labels:
        if lab == target:
            continue
        pos = lower.find(lab, content_start)
        if pos != -1:
            next_positions.append(pos)
    end = min(next_positions) if next_positions else len(text)
    return text[content_start:end].strip(" ;.\n")


def _split_items(text: str) -> list[str]:
    if not text:
        return []
    import re

    cleaned = text.replace(" and ", ", ")
    raw_parts = re.split(r"[,\n;•]+", cleaned)
    parts = []
    for chunk in raw_parts:
        item = re.sub(r"^[\\s\\-–—]+", "", chunk).strip(" .;:")
        if item:
            parts.append(item)
    return parts


def _select_draft(drafts: list[str]) -> str:
    if not drafts:
        return ""
    cleaned = [_strip_citations(d).strip() for d in drafts if d.strip()]
    return max(cleaned, key=len) if cleaned else ""


def _split_lines(text: str) -> list[str]:
    lines = [line.rstrip() for line in text.splitlines()]
    out: list[str] = []
    blank_pending = False
    for line in lines:
        cleaned = line.strip()
        if not cleaned:
            blank_pending = True
            continue
        if blank_pending and out and out[-1] != "":
            out.append("")
        blank_pending = False
        out.append(cleaned)
    return out


def _format_draft(text: str) -> str:
    import ast
    import json

    trimmed = text.strip()
    if not trimmed:
        return text
    if trimmed.startswith("{") and trimmed.endswith("}"):
        parsed = None
        try:
            parsed = json.loads(trimmed)
        except Exception:
            try:
                parsed = ast.literal_eval(trimmed)
            except Exception:
                parsed = None
        if isinstance(parsed, dict):
            lines: list[str] = []
            ordered_keys = [
                "What might be going on?",
                "What to check first?",
                "When to seek urgent care?",
            ]
            for key in ordered_keys:
                if key not in parsed:
                    continue
                lines.append(key)
                items = parsed.get(key) or []
                if isinstance(items, (list, tuple)):
                    for item in items:
                        if not item:
                            continue
                        line = str(item).strip()
                        if not line.startswith("-"):
                            line = f"- {line}"
                        lines.append(line)
                else:
                    val = str(items).strip()
                    if val:
                        lines.append(f"- {val}")
                lines.append("")
            return "\n".join(lines).strip()
    return text
