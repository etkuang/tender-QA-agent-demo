# coding: utf-8
# @Author: Wang Qingkang

import asyncio
import time
from collections.abc import Awaitable, Callable

from agent_layer.adapters.base import WebsiteSearchClient
from agent_layer.config import Settings
from agent_layer.errors import AgentError, WebsiteUnavailableError
from agent_layer.retrieval.adapter import EvidenceAdapter
from agent_layer.schemas import Evidence, ToolEvent, WebsiteQuery
from agent_layer.workflows.common import DomainProfile
from agent_layer.data_domain.schemas import DataContextBundle, DataIntentDecision

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
        context: DataContextBundle,
        profile: DomainProfile,
        requires_fresh_data: bool,
        progress_callback: ProgressCallback | None,
    ) -> tuple[list[Evidence], list[ToolEvent], list[AgentError]]:
        if not requires_fresh_data and not intent.requires_website:
            return [], [], []
        clients = [
            self.website_clients[name]
            for name in profile.website_adapters
            if name in self.website_clients
        ]
        if not clients:
            event = ToolEvent(
                stage="data_web_supplement",
                status="skipped",
                summary="该领域未配置网站补充数据源，本次只使用结构化数据库证据。",
            )
            await self._report(event, progress_callback)
            return [], [event], []

        started = time.perf_counter()
        await self._report(
            ToolEvent(
                stage="data_web_supplement",
                status="started",
                summary="正在查询网站补充来源，用于新鲜度、公开来源或数据库覆盖核验。",
            ),
            progress_callback,
        )
        query = WebsiteQuery(
            query=question,
            category=profile.category,
            keywords=[*intent.entities, *intent.business_terms],
            time_range=intent.time_range,
            region=None,
        )
        results = await asyncio.gather(
            *(client.search(query, self.settings.retrieval_batch_size) for client in clients),
            return_exceptions=True,
        )
        evidence = []
        failures = 0
        for result in results:
            if isinstance(result, Exception):
                failures += 1
                continue
            evidence.extend(
                self.evidence_adapter.from_search_result(item, profile.category)
                for item in result
            )
        status = "completed" if evidence else "failed"
        event = ToolEvent(
            stage="data_web_supplement",
            status=status,
            summary=f"网站补充检索返回 {len(evidence)} 条可用资料，{failures} 个适配器失败。",
            duration_ms=(time.perf_counter() - started) * 1000,
            details={
                "evidence_count": len(evidence),
                "failed_adapter_count": failures,
                "tables": sorted(context.table_names),
            },
        )
        await self._report(event, progress_callback)
        errors = [WebsiteUnavailableError()] if status == "failed" else []
        return evidence, [event], errors

    @staticmethod
    async def _report(event: ToolEvent, progress_callback: ProgressCallback | None) -> None:
        if progress_callback is not None:
            await progress_callback(event)