# coding: utf-8

import tempfile
import unittest
from pathlib import Path

from agent_layer.checkpoint import SQLiteCheckpointStore


class SQLiteCheckpointStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_save_load_and_delete(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteCheckpointStore(Path(directory) / "checkpoint.sqlite3")

            await store.save("run-1", "session-1", "running", {"completed_tasks": ["task-1"]})
            loaded = await store.load("run-1")
            await store.delete("run-1")

            self.assertEqual(loaded["session_id"], "session-1")
            self.assertEqual(loaded["state"]["completed_tasks"], ["task-1"])
            self.assertIsNone(await store.load("run-1"))
