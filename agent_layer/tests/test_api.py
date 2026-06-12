# coding: utf-8

import json
import os
import unittest

from fastapi.testclient import TestClient

os.environ.setdefault("LOG_SERVICE_NAME", "agent-test")

from agent_layer.api import app


class AgentApiTests(unittest.TestCase):
    def test_stream_uses_target_event_protocol(self):
        with TestClient(app) as client:
            response = client.post(
                "/chat/stream",
                headers={"X-Request-ID": "test-request"},
                json={"user_message": "你好", "history_messages": []},
            )

        self.assertEqual(response.status_code, 200)
        events = [json.loads(line) for line in response.text.splitlines()]
        event_types = [event["type"] for event in events]
        self.assertEqual(event_types[0], "route")
        self.assertIn("assistant_delta", event_types)
        self.assertEqual(event_types[-1], "final")
        self.assertIn("metrics", events[-1]["data"])

    def test_request_id_is_required(self):
        with TestClient(app) as client:
            response = client.post("/chat/stream", json={"user_message": "你好"})

        self.assertEqual(response.status_code, 400)
