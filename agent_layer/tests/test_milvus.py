# coding: utf-8

import json
import unittest

import httpx

from agent_layer.config import Settings
from agent_layer.retrieval.milvus import MilvusStore


class MilvusStoreTests(unittest.TestCase):
    def test_keyword_search_uses_sparse_bm25_field(self):
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": [
                        {
                            "id": "doc-1",
                            "distance": 3.2,
                            "content": "串通投标处罚规定",
                            "law_name": "招标投标法",
                        }
                    ],
                },
            )

        client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://milvus")
        settings = Settings(milvus_uri="http://milvus")
        store = MilvusStore(settings, client)

        results = store.search_keyword(settings.policy_collection, "串通投标", 5)

        payload = json.loads(requests[0].content)
        self.assertEqual(payload["annsField"], settings.milvus_sparse_field)
        self.assertEqual(payload["data"], ["串通投标"])
        self.assertEqual(results[0]["id"], "doc-1")

    def test_create_version_declares_bm25_function_and_hnsw(self):
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={"code": 0, "data": {}})

        client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://milvus")
        settings = Settings(milvus_uri="http://milvus")
        store = MilvusStore(settings, client)

        store.create_version("tender_qa_policy_v1")

        payload = json.loads(requests[0].content)
        self.assertEqual(payload["schema"]["functions"][0]["type"], "BM25")
        index_types = {item["indexType"] for item in payload["indexParams"]}
        self.assertEqual(index_types, {"HNSW", "SPARSE_INVERTED_INDEX"})
