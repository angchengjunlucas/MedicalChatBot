from __future__ import annotations

import argparse
import asyncio
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx


PMC_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pmc_list", required=True, help="Path to file with PMC IDs (one per line)")
    parser.add_argument("--out_dir", required=True, help="Output directory for .txt files")
    parser.add_argument("--max_concurrency", type=int, default=2)
    parser.add_argument("--min_delay_s", type=float, default=0.4)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    pmc_ids = _load_ids(Path(args.pmc_list))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    email = os.getenv("PUBMED_EMAIL")
    tool = os.getenv("PUBMED_TOOL_NAME", "medical-chatbot")
    if not email:
        raise SystemExit("Set PUBMED_EMAIL before fetching PMC full text.")

    sem = asyncio.Semaphore(args.max_concurrency)
    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = [
            _fetch_one(
                client,
                sem,
                pmc_id,
                out_dir,
                email=email,
                tool=tool,
                min_delay_s=args.min_delay_s,
            )
            for pmc_id in pmc_ids
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        failures = [r for r in results if isinstance(r, Exception)]
        if failures:
            print(f"Completed with {len(failures)} failures. Re-run to retry remaining.")


async def _fetch_one(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    pmc_id: str,
    out_dir: Path,
    email: str,
    tool: str,
    min_delay_s: float,
) -> None:
    async with sem:
        params = {
            "db": "pmc",
            "id": pmc_id,
            "retmode": "xml",
            "tool": tool,
            "email": email,
        }
        last_exc: Exception | None = None
        for attempt in range(5):
            try:
                resp = await client.get(PMC_EFETCH, params=params)
                if resp.status_code in {429, 500, 502, 503, 504}:
                    raise httpx.HTTPStatusError(
                        "retryable status",
                        request=resp.request,
                        response=resp,
                    )
                resp.raise_for_status()
                text = _extract_text(resp.text)
                if not text.strip():
                    return
                out_file = out_dir / f"{pmc_id}.txt"
                out_file.write_text(text, encoding="utf-8")
                await asyncio.sleep(min_delay_s)
                return
            except Exception as exc:
                last_exc = exc
                await asyncio.sleep(min_delay_s * (2**attempt))
                continue
        raise last_exc if last_exc else RuntimeError("Fetch failed")


def _load_ids(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return [line.strip() for line in lines if line.strip()]


def _extract_text(xml_text: str) -> str:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return ""

    title = _find_text(root, ".//article-title")
    abstract = _collect_paragraphs(root, ".//abstract//p")
    body = _collect_paragraphs(root, ".//body//p")

    parts = []
    if title:
        parts.append(title)
        parts.append("")
    if abstract:
        parts.append("Abstract:")
        parts.append(abstract)
        parts.append("")
    if body:
        parts.append("Body:")
        parts.append(body)

    return _clean("\n".join(parts))


def _find_text(root: ET.Element, path: str) -> str:
    el = root.find(path)
    return el.text.strip() if el is not None and el.text else ""


def _collect_paragraphs(root: ET.Element, path: str) -> str:
    paras = []
    for p in root.findall(path):
        txt = "".join(p.itertext()).strip()
        if txt:
            paras.append(txt)
    return "\n".join(paras)


def _clean(text: str) -> str:
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


if __name__ == "__main__":
    asyncio.run(main())
