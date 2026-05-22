"""会话管理器 - 使用 SQLChatMessageHistory 实现多轮记忆（支持过期清理和消息数量限制）"""
import time
import asyncio
import threading
from datetime import datetime
from typing import Dict, Optional, Tuple, List
from pathlib import Path

from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import BaseMessage

from config import settings


class SessionManager:
    """管理用户会话，自动处理多轮对话历史，支持过期清理和消息数量限制"""

    # 会话最大存活时间（秒），默认 7 天
    SESSION_TTL_SECONDS = getattr(settings, 'session_ttl_seconds', 7 * 24 * 3600)
    # 每个会话最大消息数（超过后自动截断，保留最近的消息）
    MAX_MESSAGES_PER_SESSION = getattr(settings, 'max_messages_per_session', 30)
    # 清理检查间隔（秒），默认 1 小时
    CLEANUP_INTERVAL_SECONDS = getattr(settings, 'cleanup_interval_seconds', 3600)

    def __init__(self):
        self._histories: Dict[str, SQLChatMessageHistory] = {}
        self._session_last_active: Dict[str, float] = {}
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_cleanup = False

        # 确保数据目录存在
        db_path = Path(settings.sqlite_db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self.connection_string = f"sqlite:///{settings.sqlite_db_path}"

        # 启动后台清理线程
        self._start_cleanup_thread()

        print(f"✅ SessionManager 初始化完成")
        print(f"   TTL: {self.SESSION_TTL_SECONDS // 3600} 小时")
        print(f"   每会话最大消息数: {self.MAX_MESSAGES_PER_SESSION}")
        print(f"   清理间隔: {self.CLEANUP_INTERVAL_SECONDS // 60} 分钟")

    def _start_cleanup_thread(self):
        """启动后台清理线程"""
        if self.CLEANUP_INTERVAL_SECONDS > 0:
            self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
            self._cleanup_thread.start()
            print("✅ 后台清理线程已启动")

    def _cleanup_loop(self):
        """后台清理循环"""
        while not self._stop_cleanup:
            time.sleep(self.CLEANUP_INTERVAL_SECONDS)
            try:
                self.cleanup_expired_sessions()
            except Exception as e:
                print(f"⚠️ 后台清理异常: {e}")

    def get_chat_history(self, session_id: str) -> SQLChatMessageHistory:
        """获取或创建会话历史，同时更新活跃时间"""
        now = time.time()
        self._session_last_active[session_id] = now

        if session_id not in self._histories:
            self._histories[session_id] = SQLChatMessageHistory(
                session_id=session_id,
                connection_string=self.connection_string,
            )
            # 启动时立即检查并截断
            self._trim_history_if_needed(session_id)

        return self._histories[session_id]


    def _trim_history_if_needed(self, session_id: str):
        """
        如果历史消息超过限制，截断到最近 N 条
        使用数据库级别的清理，避免频繁加载所有消息
        """
        try:
            history = self._histories.get(session_id)
            if not history:
                return

            messages = history.messages
            if len(messages) <= self.MAX_MESSAGES_PER_SESSION:
                return

            # 保留最近的消息
            to_keep = messages[-self.MAX_MESSAGES_PER_SESSION:]

            # 清空并重新添加
            history.clear()
            for msg in to_keep:
                if msg.type == "human":
                    history.add_user_message(msg.content)
                else:
                    history.add_ai_message(msg.content)

            print(f"✂️ 会话 {session_id[:8]}... 截断: {len(messages)} → {len(to_keep)} 条")

        except Exception as e:
            print(f"⚠️ 截断历史失败 {session_id}: {e}")



    def cleanup_expired_sessions(self) -> int:
        """清理过期会话，返回清理数量"""
        now = time.time()
        expired = [
            sid for sid, last_active in self._session_last_active.items()
            if now - last_active > self.SESSION_TTL_SECONDS
        ]

        cleaned = 0
        for sid in expired:
            try:
                if sid in self._histories:
                    # 清空历史记录
                    self._histories[sid].clear()
                    del self._histories[sid]
                if sid in self._session_last_active:
                    del self._session_last_active[sid]
                cleaned += 1
                print(f"🗑️ 清理过期会话: {sid[:8]}... (已闲置 {int((now - last_active) / 3600)} 小时)")
            except Exception as e:
                print(f"⚠️ 清理会话失败 {sid}: {e}")

        if cleaned > 0:
            print(f"📊 本次清理 {cleaned} 个过期会话，当前活跃会话: {len(self._session_last_active)}")

        return cleaned

    def get_context(self, session_id: str, max_messages: int = 10) -> str:
        """获取格式化的历史上下文，用于注入 Prompt"""
        if not session_id:
            return ""

        # 先清理过期会话
        self.cleanup_expired_sessions()

        history = self.get_chat_history(session_id)
        # 使用 max_messages 参数限制（调用方可控制）
        messages = history.messages[-max_messages:] if max_messages > 0 else history.messages

        if not messages:
            return ""

        context_lines = []
        for msg in messages:
            role = "用户" if msg.type == "human" else "助手"
            content = msg.content[:800]  # 截断过长内容
            context_lines.append(f"{role}: {content}")

        return "\n".join(context_lines)

    def get_full_history(self, session_id: str, max_messages: int = 50) -> List[Dict]:
        """获取完整的对话历史（用于调试）"""
        if not session_id:
            return []

        history = self.get_chat_history(session_id)
        messages = history.messages[-max_messages:] if max_messages > 0 else history.messages

        result = []
        for msg in messages:
            result.append({
                "role": "user" if msg.type == "human" else "assistant",
                "content": msg.content,
                "timestamp": getattr(msg, "additional_kwargs", {}).get("timestamp", "")
            })
        return result

    def add_user_message(self, session_id: str, message: str):
        """记录用户消息"""
        if not session_id:
            return
        history = self.get_chat_history(session_id)
        history.add_user_message(message)
        # 添加后检查是否需要截断
        self._trim_history_if_needed(session_id)

    def add_ai_message(self, session_id: str, message: str):
        """记录 AI 回复"""
        if not session_id:
            return
        history = self.get_chat_history(session_id)
        history.add_ai_message(message)
        # 添加后检查是否需要截断
        self._trim_history_if_needed(session_id)

    def clear_session(self, session_id: str):
        """清除指定会话"""
        if session_id in self._histories:
            try:
                self._histories[session_id].clear()
            except Exception as e:
                print(f"⚠️ 清空会话历史失败: {e}")
            del self._histories[session_id]
        if session_id in self._session_last_active:
            del self._session_last_active[session_id]
        print(f"🗑️ 会话已清除: {session_id[:8]}...")

    def get_session_info(self, session_id: str) -> Dict:
        """获取会话信息（用于调试）"""
        if session_id not in self._session_last_active:
            return {"exists": False}

        history = self._histories.get(session_id)
        message_count = len(history.messages) if history else 0
        last_active = self._session_last_active.get(session_id, 0)
        idle_hours = (time.time() - last_active) / 3600 if last_active else 0

        return {
            "exists": True,
            "session_id": session_id[:8] + "...",
            "message_count": message_count,
            "last_active": datetime.fromtimestamp(last_active).isoformat() if last_active else None,
            "idle_hours": round(idle_hours, 1),
            "ttl_hours": self.SESSION_TTL_SECONDS / 3600
        }

    def get_stats(self) -> Dict:
        """获取统计信息"""
        self.cleanup_expired_sessions()
        return {
            "active_sessions": len(self._session_last_active),
            "cached_histories": len(self._histories),
            "max_messages_per_session": self.MAX_MESSAGES_PER_SESSION,
            "ttl_hours": self.SESSION_TTL_SECONDS // 3600,
            "cleanup_interval_minutes": self.CLEANUP_INTERVAL_SECONDS // 60
        }

    def stop(self):
        """停止后台清理线程（用于优雅关闭）"""
        self._stop_cleanup = True
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
        print("✅ SessionManager 已停止")

    def get_recent_dialogues(self, session_id: str, max_pairs: int = 2) -> str:
        """获取最近 N 轮对话（用户+助手成对），用于问题重写"""
        if not session_id:
            return ""

        history = self.get_chat_history(session_id)
        messages = history.messages

        if not messages:
            return ""

        # 取最近的消息（最多 max_pairs * 2 条）
        recent = messages[-max_pairs * 2:] if max_pairs > 0 else messages

        lines = []
        for msg in recent:
            role = "用户" if msg.type == "human" else "助手"
            lines.append(f"{role}: {msg.content}")

        return "\n".join(lines)