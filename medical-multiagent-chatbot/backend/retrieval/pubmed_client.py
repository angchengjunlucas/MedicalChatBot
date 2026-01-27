from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class PubMedArticle:
    pmid: str
    title: str
    year: int | None
    journal: str
    url: str
    authors: list[str]
    abstract: str | None = None


class PubMedClient:
    def __init__(
        self,
        email: str,
        tool: str,
        api_key: str | None = None,
        base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
        timeout_s: float = 20.0,
        max_retries: int = 2,
        cache_ttl_s: int = 3600,
    ) -> None:
        self._email = email
        self._tool = tool
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._cache_ttl_s = cache_ttl_s
        self._cache: dict[str, tuple[float, Any]] = {}
        self._lock = asyncio.Lock()

    async def search(self, query: str, retmax: int = 5) -> list[str]:
        params = self._with_common_params(
            {
                "db": "pubmed",
                "term": query,
                "retmax": retmax,
                "retmode": "json",
            }
        )
        data = await self._get_json("esearch.fcgi", params)
        return data.get("esearchresult", {}).get("idlist", [])

    async def summary(self, pmids: list[str]) -> list[PubMedArticle]:
        if not pmids:
            return []
        params = self._with_common_params(
            {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "json",
            }
        )
        data = await self._get_json("esummary.fcgi", params)
        result = data.get("result", {})
        uids = result.get("uids", [])
        articles: list[PubMedArticle] = []
        for uid in uids:
            item = result.get(uid, {})
            title = item.get("title", "")
            journal = item.get("fulljournalname") or item.get("source") or ""
            pubdate = item.get("pubdate", "")
            year = self._extract_year(pubdate)
            authors = [a.get("name", "") for a in item.get("authors", []) if a.get("name")]
            url = f"https://pubmed.ncbi.nlm.nih.gov/{uid}/"
            articles.append(
                PubMedArticle(
                    pmid=str(uid),
                    title=title,
                    year=year,
                    journal=journal,
                    url=url,
                    authors=authors,
                    abstract=None,
                )
            )
        return articles

    async def search_and_summary(self, query: str, retmax: int = 5) -> list[PubMedArticle]:
        pmids = await self.search(query, retmax=retmax)
        return await self.summary(pmids)

    def _with_common_params(self, params: dict[str, Any]) -> dict[str, Any]:
        params["tool"] = self._tool
        params["email"] = self._email
        if self._api_key:
            params["api_key"] = self._api_key
        return params

    async def _get_json(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        cache_key = self._cache_key(path, params)
        cached = await self._cache_get(cache_key)
        if cached is not None:
            return cached

        url = f"{self._base_url}/{path}"
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                    resp = await client.get(url, params=params)
                    if resp.status_code in {429, 500, 502, 503, 504}:
                        raise httpx.HTTPStatusError(
                            "retryable status",
                            request=resp.request,
                            response=resp,
                        )
                    resp.raise_for_status()
                    data = resp.json()
                    await self._cache_set(cache_key, data)
                    return data
            except Exception as exc:  # pragma: no cover - exercised by retry behavior
                last_exc = exc
                if attempt < self._max_retries:
                    await asyncio.sleep(0.5 * (2**attempt))
                    continue
                break
        raise RuntimeError("PubMed request failed") from last_exc

    async def _cache_get(self, key: str) -> Any | None:
        async with self._lock:
            item = self._cache.get(key)
            if not item:
                return None
            ts, value = item
            if time.time() - ts > self._cache_ttl_s:
                self._cache.pop(key, None)
                return None
            return value

    async def _cache_set(self, key: str, value: Any) -> None:
        async with self._lock:
            self._cache[key] = (time.time(), value)

    @staticmethod
    def _extract_year(pubdate: str) -> int | None:
        for part in pubdate.split():
            if part.isdigit() and len(part) == 4:
                return int(part)
        return None

    @staticmethod
    def _cache_key(path: str, params: dict[str, Any]) -> str:
        parts = [path] + [f"{k}={params[k]}" for k in sorted(params)]
        return "|".join(parts)

"""
given a medical question, it returns pubmed ids and metadata for relevant paper
1. Search --> IDs of paper that match this question
2. Summary --> For those IDs, give me the paper details

"""