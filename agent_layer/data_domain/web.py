# coding: utf-8
# @Author: Wang Qingkang

import time
from collections.abc import Awaitable, Callable

from agent_layer.adapters.base import WebsiteSearchClient
from agent_layer.config import Settings
from agent_layer.errors import AgentError, WebsiteUnavailableError
from agent_layer.retrieval.adapter import EvidenceAdapter
from agent_layer.schemas import Category, Evidence, ToolEvent, WebsiteQuery
from agent_layer.data_domain.schemas import DataIntentDecision

ProgressCallback = Callable[[ToolEvent], Awaitable[None]]


class WebsiteSupplementer:
    def __init__(
        self,
        evidence_adapter: EvidenceAdapter,
        website_clients: dict[str, WebsiteSearchClient],
        settings: Settings,
    ):
        self.evidence_adapter = evidence_adapter
        self.website_clients = website_clients
        self.settings = settings

    async def search(
        self,
        question: str,
        intent: DataIntentDecision,
        category: Category,
        progress_callback: ProgressCallback | None,
    ) -> tuple[list[Evidence], list[ToolEvent], list[AgentError]]:
        client = self.website_clients.get(category.value)
        if client is None:
            event = ToolEvent(
                stage="website",
                status="skipped",
                summary=f"No website adapter is configured for {category.value}.",
            )
            await self._report(event, progress_callback)
            return [], [event], []

        start_event = ToolEvent(
            stage="website",
            status="started",
            summary=f"Searching website evidence for {category.value}.",
        )
        await self._report(start_event, progress_callback)
        started = time.perf_counter()
        query = WebsiteQuery(
            query=question,
            category=category,
            keywords=[*intent.entities, *intent.business_terms],
            time_range=intent.time_range,
            region=None,
        )
        try:
            results = await client.search(query, self.settings.retrieval_batch_size)
        except Exception:
            event = ToolEvent(
                stage="website",
                status="failed",
                summary=f"Website adapter for {category.value} is temporarily unavailable.",
                duration_ms=(time.perf_counter() - started) * 1000,
            )
            await self._report(event, progress_callback)
            return [], [start_event, event], [WebsiteUnavailableError()]

        evidence = [
            self.evidence_adapter.from_search_result(item, category)
            for item in results
        ]
        status = "completed" if evidence else "failed"
        event = ToolEvent(
            stage="website",
            status=status,
            summary=f"Website search returned {len(evidence)} evidence items for {category.value}.",
            duration_ms=(time.perf_counter() - started) * 1000,
            details={"evidence_count": len(evidence), "category": category.value},
        )
        await self._report(event, progress_callback)
        errors = [WebsiteUnavailableError()] if status == "failed" else []
        return evidence, [start_event, event], errors

    @staticmethod
    async def _report(event: ToolEvent, progress_callback: ProgressCallback | None) -> None:
        if progress_callback is not None:
            await progress_callback(event)