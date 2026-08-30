# coding: utf-8
# @Author: Wang Qingkang

import httpx

from common.api_contracts.knowledge_base_api import KnowledgeSearchRequest, KnowledgeSearchResponse
from common.logger import get_request_id
from agent_layer_old.config import Settings


class KnowledgeBaseClient:
    def __init__(self, settings: Settings):
        self.base_url = settings.knowledge_base_url.rstrip("/")
        self.timeout = settings.knowledge_base_timeout_seconds

    async def search(self, request: KnowledgeSearchRequest) -> KnowledgeSearchResponse:
        headers = {"X-Request-ID": get_request_id()}
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
            response = await client.post(
                "/search",
                json=request.model_dump(mode="json"),
                headers=headers,
            )
        response.raise_for_status()
        return KnowledgeSearchResponse.model_validate(response.json())