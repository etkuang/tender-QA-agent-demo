# coding: utf-8
# @Author: Wang Qingkang

import time
import aiosqlite

from app_backend_layer.config import settings


class HistoryManager:
    def __init__(self):
        self.db_dir = settings.db_dir
        self.db_path = settings.db_path
        self.conn: aiosqlite.Connection | None = None

    async def connect(self):
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.conn = await aiosqlite.connect(self.db_path)
        self.conn.row_factory = aiosqlite.Row  # Configure connection to return dictionary-like row objects
        await self.conn.execute("PRAGMA journal_mode=WAL;")  # allows simultaneous readers and writers
        await self.conn.execute("PRAGMA synchronous=NORMAL;")  # balancing I/O performance and data safety
        await self.conn.execute("PRAGMA foreign_keys=ON;")  # Enforce foreign key constraints
        await self.conn.commit()

    async def disconnect(self):
        if self.conn:
            await self.conn.close()
            self.conn = None

    async def initialize_db(self):
        # table 1: session data
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT PRIMARY KEY, 
                title TEXT NOT NULL, 
                updated_at REAL NOT NULL
            )
        """)
        # table 2: message data
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                type TEXT NOT NULL, 
                content TEXT NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY(session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE
            )
        """)
        await self.conn.execute("CREATE INDEX IF NOT EXISTS idx_updated_at ON chat_sessions (updated_at DESC)")
        await self.conn.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON chat_messages (session_id)")
        await self.conn.commit()

    async def get_all_metadata(self):
        async with self.conn.execute("SELECT session_id, title FROM chat_sessions ORDER BY updated_at DESC") as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def load_messages(self, session_id: str):
        query = "SELECT type, content FROM chat_messages WHERE session_id = ? ORDER BY created_at ASC"

        async with self.conn.execute(query, (session_id,)) as cursor:
            rows = await cursor.fetchall()
            return [{"type": row["type"], "content": row["content"]} for row in rows]

    async def append_message(self, session_id: str, title: str, msg_type: str, content: str):
        current_time = time.time()

        session_query = """
            INSERT INTO chat_sessions (session_id, title, updated_at) 
            VALUES (?, ?, ?) 
            ON CONFLICT(session_id) DO UPDATE SET 
                title = excluded.title, 
                updated_at = excluded.updated_at
        """
        msg_query = """
            INSERT INTO chat_messages (session_id, type, content, created_at)
            VALUES (?, ?, ?, ?)
        """

        try:
            await self.conn.execute(session_query, (session_id, title, current_time))
            await self.conn.execute(msg_query, (session_id, msg_type, content, current_time))
            await self.conn.commit()
        except Exception as e:
            await self.conn.rollback()
            raise e

    async def delete_session(self, session_id):
        await self.conn.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
        await self.conn.commit()