# coding: utf-8
# @Author: Wang Qingkang

import time
from collections.abc import Awaitable, Callable

from agent_layer_old.adapters.base import WebsiteSearchClient
from agent_layer_old.config import Settings
from agent_layer_old.errors import AgentError, WebsiteUnavailableError
from agent_layer_old.retrieval.adapter import EvidenceAdapter
from agent_layer_old.schemas import Category, Evidence, SourceTier, ToolEvent, WebsiteQuery


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
        query: str,
        category: Category,
        keywords: list[str],
        required_source_tier: SourceTier | None,
        progress_callback: Callable[[ToolEvent], Awaitable[None]],
    ) -> tuple[list[Evidence], list[ToolEvent], list[AgentError]]:
        client = self.website_clients.get(category.value)
        if client is None:
            event = ToolEvent(
                stage="website",
                status="skipped",
                summary="未配置当前类别的网站数据源。",
            )
            await progress_callback(event)
            return [], [event], []

        start_event = ToolEvent(
            stage="website",
            status="started",
            summary="正在检索当前类别的网站证据。",
        )
        await progress_callback(start_event)
        started = time.perf_counter()
        website_query = WebsiteQuery(
            query=query,
            category=category,
            keywords=keywords,
        )
        try:
            results = await client.search(
                website_query,
                self.settings.retrieval_batch_size,
            )
        except Exception:
            event = ToolEvent(
                stage="website",
                status="failed",
                summary="当前类别的网站数据源暂时不可用。",
                duration_ms=(time.perf_counter() - started) * 1000,
            )
            await progress_callback(event)
            return [], [start_event, event], [WebsiteUnavailableError()]

        if required_source_tier is not None:
            results = [
                item
                for item in results
                if item.source_tier == required_source_tier
            ]
        evidence = [
            self.evidence_adapter.from_search_result(item, category)
            for item in results
        ]
        status = "completed" if evidence else "failed"
        event = ToolEvent(
            stage="website",
            status=status,
            summary=f"网站检索返回 {len(evidence)} 条证据。",
            duration_ms=(time.perf_counter() - started) * 1000,
            details={"evidence_count": len(evidence), "category": category.value},
        )
        await progress_callback(event)
        errors = [WebsiteUnavailableError()] if status == "failed" else []
        return evidence, [start_event, event], errors
