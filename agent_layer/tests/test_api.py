# coding: utf-8

import json
import os
import unittest

from fastapi.testclient import TestClient

os.environ.setdefault("LOG_SERVICE_NAME", "agent-test")

from agent_layer.api import app


class AgentApiTests(unittest.TestCase):
    def test_stream_uses_backend_transport_protocol(self):
        with TestClient(app) as client:
            response = client.post(
                "/chat/stream",
                headers={"X-Request-ID": "test-request"},
                json={"user_message": "你好", "history_messages": []},
            )

        self.assertEqual(response.status_code, 200)
        events = [json.loads(line) for line in response.text.splitlines()]
        event_types = [event["type"] for event in events]
        self.assertTrue(set(event_types).issubset({"assistant", "reasoning", "error"}))
        self.assertEqual(event_types[0], "reasoning")
        self.assertIn("assistant", event_types)
        reasoning = "".join(event["content"] for event in events if event["type"] == "reasoning")
        self.assertIn("正在理解您的问题", reasoning)
        self.assertIn("会话控制消息", reasoning)

    def test_request_id_is_required(self):
        with TestClient(app) as client:
            response = client.post("/chat/stream", json={"user_message": "你好"})

        self.assertEqual(response.status_code, 400)
