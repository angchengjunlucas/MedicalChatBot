from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.llm.local_provider import LocalPeftProvider
from backend.llm.router import ModelRouter
from backend.safety.postprocess import safety_postprocess
from backend.storage.db import init_db
from backend.retrieval.embeddings import OpenAIEmbeddingsProvider
from backend.retrieval.vectorstore import ChromaVectorStore
from backend.retrieval.pipeline import KBRetrievalPipeline, RAGPipeline
from backend.retrieval.pubmed_client import PubMedClient
from backend.retrieval.rewrite import QueryRewriter
from backend.orchestration.runner import Orchestrator


@dataclass
class TestResult:
    name: str
    status: str
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_base_model", default=os.getenv("LOCAL_BASE_MODEL", ""))
    parser.add_argument("--local_supervisor_adapter", default=os.getenv("LOCAL_SUPERVISOR_ADAPTER", ""))
    parser.add_argument("--local_cardiology_adapter", default=os.getenv("LOCAL_CARDIOLOGY_ADAPTER", ""))
    parser.add_argument("--local_geriatrics_adapter", default=os.getenv("LOCAL_GERIATRICS_ADAPTER", ""))
    parser.add_argument("--local_mental_adapter", default=os.getenv("LOCAL_MENTAL_ADAPTER", ""))
    parser.add_argument("--base_url", default=os.getenv("LLM_API_BASE", ""))
    parser.add_argument("--api_key", default=os.getenv("LLM_API_KEY", ""))
    parser.add_argument("--embedding_model", default=os.getenv("EMBEDDING_MODEL", ""))
    parser.add_argument("--chroma_path", default=os.getenv("CHROMA_PATH", "./data/chroma"))
    parser.add_argument("--pubmed_email", default=os.getenv("PUBMED_EMAIL", ""))
    parser.add_argument("--pubmed_tool", default=os.getenv("PUBMED_TOOL_NAME", ""))
    parser.add_argument("--log_db_url", default=os.getenv("LOG_DB_URL", "sqlite:///./data/logs.db"))
    parser.add_argument("--max_tokens", type=int, default=96)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    results: list[TestResult] = []

    # Safety postprocess
    try:
        sample = "You have pneumonia and should take amoxicillin 500mg twice daily."
        safe = safety_postprocess(sample)
        results.append(
            TestResult(
                name="safety_postprocess",
                status="PASS",
                detail=f"flags={safe.flags}",
            )
        )
    except Exception as exc:
        results.append(TestResult("safety_postprocess", "FAIL", str(exc)))

    # Logging init
    try:
        init_db(args.log_db_url)
        results.append(TestResult("logging_init", "PASS", args.log_db_url))
    except Exception as exc:
        results.append(TestResult("logging_init", "FAIL", str(exc)))

    # Local adapters
    adapter_paths = {
        "supervisor": args.local_supervisor_adapter,
        "cardiology": args.local_cardiology_adapter,
        "geriatrics": args.local_geriatrics_adapter,
        "mental": args.local_mental_adapter,
    }
    adapter_paths = {k: v for k, v in adapter_paths.items() if v}

    provider = None
    if args.local_base_model and adapter_paths:
        try:
            provider = LocalPeftProvider(
                base_model_id=args.local_base_model,
                adapter_paths=adapter_paths,
            )
            prompts = {
                "cardiology": "Briefly explain what an ECG measures.",
                "geriatrics": "List two common concerns in older adults.",
                "mental": "Give a supportive, general tip for managing anxiety.",
            }
            for role, prompt in prompts.items():
                resp = await provider.generate_with_role(
                    role,
                    [{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=args.max_tokens,
                )
                if not resp.content.strip():
                    raise RuntimeError(f"empty response for role {role}")
            results.append(TestResult("local_adapters", "PASS", "responses generated"))
        except Exception as exc:
            results.append(TestResult("local_adapters", "FAIL", str(exc)))
    else:
        results.append(
            TestResult(
                "local_adapters",
                "SKIP",
                "missing LOCAL_BASE_MODEL or adapter paths",
            )
        )

    # KB retrieval
    kb_pipeline = None
    if args.base_url and args.embedding_model and Path(args.chroma_path).exists():
        try:
            embedder = OpenAIEmbeddingsProvider(
                base_url=args.base_url,
                api_key=args.api_key or "ollama",
                model_id=args.embedding_model,
            )
            store = ChromaVectorStore(path=args.chroma_path)
            kb_pipeline = KBRetrievalPipeline(vector_store=store, embedder=embedder)
            kb_result = await kb_pipeline.retrieve_hybrid("heart failure", top_k=3)
            results.append(
                TestResult(
                    "kb_retrieval",
                    "PASS",
                    f"records={len(kb_result.records)}",
                )
            )
        except Exception as exc:
            results.append(TestResult("kb_retrieval", "FAIL", str(exc)))
    else:
        results.append(
            TestResult(
                "kb_retrieval",
                "SKIP",
                "missing LLM_API_BASE/EMBEDDING_MODEL or CHROMA_PATH",
            )
        )

    # PubMed retrieval
    pubmed_client = None
    if args.pubmed_email and args.pubmed_tool:
        try:
            pubmed_client = PubMedClient(email=args.pubmed_email, tool=args.pubmed_tool)
            articles = await pubmed_client.search_and_summary("heart failure", retmax=2)
            results.append(
                TestResult(
                    "pubmed_retrieval",
                    "PASS",
                    f"articles={len(articles)}",
                )
            )
        except Exception as exc:
            results.append(TestResult("pubmed_retrieval", "FAIL", str(exc)))
    else:
        results.append(
            TestResult(
                "pubmed_retrieval",
                "SKIP",
                "missing PUBMED_EMAIL/PUBMED_TOOL_NAME",
            )
        )

    # End-to-end orchestrator (requires local provider + KB + PubMed)
    if provider and kb_pipeline and pubmed_client:
        try:
            router = ModelRouter(
                provider=provider,
                default_model="local",
                supervisor_model="supervisor",
                cardiology_model="cardiology",
                geriatrics_model="geriatrics",
                mental_model="mental",
            )
            rewriter = QueryRewriter(router)
            rag = RAGPipeline(kb_pipeline=kb_pipeline, pubmed_client=pubmed_client, rewriter=rewriter)
            orchestrator = Orchestrator(rag=rag, router=router)
            result = await orchestrator.run("Older adult with chest pain and dizziness.")
            if not result.final_text.strip():
                raise RuntimeError("empty final response")
            results.append(TestResult("orchestrator_chat", "PASS", "final_text generated"))
        except Exception as exc:
            results.append(TestResult("orchestrator_chat", "FAIL", str(exc)))
    else:
        results.append(
            TestResult(
                "orchestrator_chat",
                "SKIP",
                "requires local adapters + KB + PubMed",
            )
        )

    print(json.dumps([r.__dict__ for r in results], indent=2))


if __name__ == "__main__":
    asyncio.run(main())
