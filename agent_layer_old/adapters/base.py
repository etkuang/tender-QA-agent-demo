# coding: utf-8

import asyncio
import hashlib
import json
import time
from abc import ABC, abstractmethod
from typing import Protocol

import httpx
from pydantic import BaseModel, Field

from agent_layer_old.schemas import Category, SearchResult, SourceTier, WebsiteQuery


class WebsiteSearchClient(Protocol):
    async def search(
        self,
        query: WebsiteQuery,
        limit: int,
    ) -> list[SearchResult]: ...


class WebsiteAdapterConfig(BaseModel):
    name: str
    base_url: str
    source_tier: SourceTier
    timeout_seconds: float = 15.0
    max_retries: int = 2
    cache_ttl_seconds: int = 300
    headers: dict[str, str] = Field(default_factory=dict, exclude=True)


class WebsiteRequest(BaseModel):
    method: str = "GET"
    path: str
    params: dict = Field(default_factory=dict)
    json_body: dict | None = None


class BaseWebsiteAdapter(ABC):
    expected_category: Category

    def __init__(self, config: WebsiteAdapterConfig):
        self.config = config
        self.client = httpx.AsyncClient(
            base_url=config.base_url.rstrip("/"),
            headers=config.headers,
            timeout=config.timeout_seconds,
            follow_redirects=True,
        )
        self.cache = {}

    async def search(
        self,
        query: WebsiteQuery,
        limit: int,
    ) -> list[SearchResult]:
        if query.category != self.expected_category:
            raise ValueError(f"Adapter {self.config.name} cannot serve category {query.category.value}")
        cache_key = self._cache_key(query, limit)
        cached = self.cache.get(cache_key)
        if cached and cached[0] > time.monotonic():
            return cached[1]

        request = self.build_request(query, limit)
        response = await self._send(request)
        results = self.parse_response(response, query, limit)
        normalized = [self._normalize_result(result) for result in results[:limit]]
        self.cache[cache_key] = (time.monotonic() + self.config.cache_ttl_seconds, normalized)
        return normalized

    @abstractmethod
    def build_request(
        self,
        query: WebsiteQuery,
        limit: int,
    ) -> WebsiteRequest: ...

    @abstractmethod
    def parse_response(
        self,
        response: httpx.Response,
        query: WebsiteQuery,
        limit: int,
    ) -> list[SearchResult]: ...

    async def _send(self, request: WebsiteRequest) -> httpx.Response:
        last_error = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = await self.client.request(
                    request.method,
                    request.path,
                    params=request.params,
                    json=request.json_body,
                )
                if response.status_code >= 500 and attempt < self.config.max_retries:
                    await asyncio.sleep(0.5 * (2 ** attempt))
                    continue
                response.raise_for_status()
                return response
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    raise
                await asyncio.sleep(0.5 * (2 ** attempt))
        raise last_error

    def _normalize_result(self, result: SearchResult) -> SearchResult:
        content_hash = result.content_hash or hashlib.sha256(
            f"{result.url}|{result.title}|{result.snippet}".encode("utf-8")
        ).hexdigest()
        return result.model_copy(
            update={
                "source_name": self.config.name,
                "source_tier": self.config.source_tier,
                "content_hash": content_hash,
            }
        )

    def _cache_key(self, query: WebsiteQuery, limit: int) -> str:
        payload = {
            "adapter": self.config.name,
            "query": query.model_dump(mode="json"),
            "limit": limit,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
