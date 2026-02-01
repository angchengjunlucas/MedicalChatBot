from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from backend.retrieval.embeddings import OpenAIEmbeddingsProvider
from backend.retrieval.ingest import TextDocument, ingest_documents
from backend.retrieval.vectorstore import ChromaVectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Folder with .txt/.md/.pdf files")
    parser.add_argument("--chroma_path", required=True)
    parser.add_argument("--collection", default="kb_store")
    parser.add_argument("--max_chars", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_files", type=int, default=0, help="Limit number of files (0 = all)")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    base_url = _get_env("LLM_API_BASE")
    api_key = _get_env("LLM_API_KEY")
    model_id = _get_env("EMBEDDING_MODEL")

    embedder = OpenAIEmbeddingsProvider(
        base_url=base_url,
        api_key=api_key,
        model_id=model_id,
    )
    store = ChromaVectorStore(path=args.chroma_path, collection_name=args.collection)

    files = [
        p
        for p in sorted(input_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in {".txt", ".md", ".pdf"}
    ]
    if args.max_files and args.max_files > 0:
        files = files[: args.max_files]

    docs: list[TextDocument] = []
    for path in files:
        text = _read_file(path)
        if not text.strip():
            continue
        docs.append(
            TextDocument(
                doc_id=str(path.relative_to(input_dir)),
                text=text,
                metadata={"source": str(path)},
            )
        )

    total = await ingest_documents(
        docs,
        embedder=embedder,
        store=store,
        max_chars=args.max_chars,
        overlap=args.overlap,
        batch_size=args.batch_size,
    )
    print(f"Ingested {total} chunks from {len(docs)} documents.")


def _read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except Exception as exc:
            raise SystemExit("pypdf is required for PDF ingestion. Install it and retry.") from exc
        reader = PdfReader(str(path))
        return "\n".join([p.extract_text() or "" for p in reader.pages])
    return path.read_text(encoding="utf-8", errors="ignore")


def _get_env(name: str) -> str:
    value = __import__("os").getenv(name)
    if not value:
        raise SystemExit(f"Missing environment variable: {name}")
    return value


if __name__ == "__main__":
    asyncio.run(main())
