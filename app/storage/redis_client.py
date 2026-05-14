"""Redis客户端"""

import redis.asyncio as redis
from typing import Optional

from app.config import settings


class RedisClient:
    _instance = None
    _client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def _get_client(self):
        if self._client is None:
            self._client = redis.Redis(
                host=settings.redis_host, port=settings.redis_port,
                db=settings.redis_db, decode_responses=True, socket_timeout=5
            )
            print(f"Redis connected: {settings.redis_host}:{settings.redis_port}")
        return self._client

    async def get(self, key: str) -> Optional[str]:
        client = await self._get_client()
        return await client.get(key)

    async def set(self, key: str, value: str, ttl: int = None):
        client = await self._get_client()
        if ttl:
            await client.setex(key, ttl, value)
        else:
            await client.set(key, value)

    async def delete(self, key: str):
        client = await self._get_client()
        await client.delete(key)

    async def exists(self, key: str) -> bool:
        client = await self._get_client()
        return await client.exists(key) > 0

    async def close(self):
        if self._client:
            await self._client.close()


redis_client = RedisClient()
