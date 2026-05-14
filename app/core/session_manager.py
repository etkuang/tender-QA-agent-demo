"""会话管理器"""

import json
import re
import uuid
from typing import Dict, List, Optional, Tuple

from app.storage.redis_client import redis_client
from app.config import settings


class SessionManager:
    """会话管理器"""

    def __init__(self):
        self.redis = redis_client
        self.ttl = settings.session_ttl
        self.max_history = settings.max_history

    async def get_or_create_session(self, session_id: Optional[str] = None) -> Tuple[str, bool]:
        if session_id and await self.session_exists(session_id):
            return session_id, False
        new_id = f"session_{uuid.uuid4().hex[:16]}"
        return new_id, True

    async def session_exists(self, session_id: str) -> bool:
        key = f"session:{session_id}"
        return await self.redis.exists(key)

    async def add_turn(self, session_id: str, question: str, answer: str, entities: Dict = None):
        key = f"session:{session_id}"
        session_data = await self.get_session(session_id) or {}
        history = session_data.get("history", [])
        history.append({
            "question": question, "answer": answer,
            "entities": entities or {}, "timestamp": self._now_iso()
        })
        if len(history) > self.max_history:
            history = history[-self.max_history:]
        session_data = {
            "history": history, "last_question": question,
            "last_answer": answer, "last_entities": entities or {},
            "updated_at": self._now_iso()
        }
        await self.redis.set(key, json.dumps(session_data, ensure_ascii=False), self.ttl)

    async def get_session(self, session_id: str) -> Optional[Dict]:
        key = f"session:{session_id}"
        data = await self.redis.get(key)
        return json.loads(data) if data else None

    async def get_history(self, session_id: str, last_n: int = 3) -> List[Dict]:
        session = await self.get_session(session_id)
        return session.get("history", [])[-last_n:] if session else []

    async def rewrite_query_with_context(self, session_id: str, query: str) -> str:
        session = await self.get_session(session_id)
        if not session:
            return query
        last_entities = session.get("last_entities", {})
        project_name = last_entities.get("project_name") or last_entities.get("title", "")
        if project_name:
            if "那个项目" in query:
                query = query.replace("那个项目", project_name)
            if "该项目" in query:
                query = query.replace("该项目", project_name)
            if "它" in query and len(query) < 20:
                query = query.replace("它", project_name)
        return query

    async def extract_entities(self, question: str, answer: str, retrieved_context: List) -> Dict:
        entities = {}
        for ctx in retrieved_context[:1]:
            data = ctx.get("data", {})
            if data.get("project_name"):
                entities["project_name"] = data["project_name"]
            if data.get("winner"):
                entities["winner"] = data["winner"]
        match = re.search(r'【([^】]+)】', answer)
        if match and not entities.get("project_name"):
            entities["project_name"] = match.group(1)
        return entities

    async def delete_session(self, session_id: str):
        key = f"session:{session_id}"
        await self.redis.delete(key)

    def _now_iso(self) -> str:
        from datetime import datetime
        return datetime.now().isoformat()
