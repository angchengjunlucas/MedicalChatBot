import pytest

from backend.retrieval.pipeline import KBRetrievalPipeline, RAGPipeline
from backend.retrieval.vectorstore import VectorRecord


@pytest.mark.asyncio
async def test_rag_pipeline_returns_context_bundle() -> None:
    class FakeKB(KBRetrievalPipeline):
        async def retrieve(self, query: str, top_k: int = 5):
            return type(
                "KBRetrievalResult",
                (),
                {"records": [VectorRecord(doc_id="d", chunk_id="c", text="t", metadata={}, score=0.9)]},
            )()

        async def retrieve_hybrid(self, query: str, top_k: int = 5):
            return await self.retrieve(query, top_k=top_k)

    class FakePubMed:
        async def search_and_summary(self, query: str, retmax: int = 5, **kwargs):
            return []

    rag = RAGPipeline(kb_pipeline=FakeKB(None, None), pubmed_client=FakePubMed())
    bundle = await rag.retrieve("heart failure", top_k=1, pubmed_k=1)

    assert bundle.query == "heart failure"
    assert len(bundle.kb_passages) == 1

# test to see if my RAG pipeline returns a context bundle, which is the localDB and the extra files taken on the website
