# coding: utf-8

import unittest

from agent_layer.config import Settings
from agent_layer.retrieval.adapter import EvidenceAdapter
from agent_layer.retrieval.parent_context import ParentContextExpander
from agent_layer.retrieval.pipeline import RetrievalPipeline


class FakeStore:
    def get_by_ids(self, collection: str, document_ids: list[str]) -> list[dict]:
        return [
            {
                "id": "parent-law-68",
                "text": "第六十八条 完整法条内容",
                "data": {
                    "chunk_type": "parent",
                    "law_name": "中华人民共和国招标投标法",
                    "article_id": 68,
                    "source": "法规全书",
                },
                "score": None,
            }
        ]


class FakeRetriever:
    def __init__(self):
        self.store = FakeStore()

    def search_article_exact(self, law_name: str, article_num: str, as_of_date: str | None = None) -> list[dict]:
        return []

    def search(
        self,
        query: str,
        collection: str,
        top_k: int | None = None,
        scalar_filter: str = "",
    ) -> list[dict]:
        return [
            {
                "id": "child-law-68-0",
                "text": "第六十八条 片段",
                "data": {
                    "chunk_type": "child",
                    "parent_id": "parent-law-68",
                    "law_name": "中华人民共和国招标投标法",
                    "article_id": 68,
                },
                "score": 0.9,
                "vector_rank": 1,
                "bm25_rank": 2,
                "fusion_score": 0.8,
            }
        ]

    @staticmethod
    def build_policy_filter(as_of_date: str | None = None, region: str | None = None) -> str:
        return 'validity_status == "effective"'


class RetrievalPipelineTests(unittest.TestCase):
    def test_child_hit_is_replaced_with_parent_evidence(self):
        retriever = FakeRetriever()
        pipeline = RetrievalPipeline(
            retriever,
            EvidenceAdapter(),
            Settings(),
            ParentContextExpander(retriever.store),
        )

        result = pipeline.retrieve_policy("弄虚作假骗取中标如何处罚？")

        self.assertEqual(len(result.evidence), 1)
        self.assertIn("完整法条内容", result.evidence[0].content)
        self.assertEqual(result.evidence[0].article_id, "68")

    def test_same_article_number_from_different_laws_remains_distinct(self):
        adapter = EvidenceAdapter()
        first = adapter.from_policy_chunk(
            {
                "id": "law-a-1",
                "text": "法律A第一条",
                "data": {"law_name": "法律A", "article_id": 1, "chunk_type": "parent"},
                "score": 0.8,
            }
        )
        second = adapter.from_policy_chunk(
            {
                "id": "law-b-1",
                "text": "法律B第一条",
                "data": {"law_name": "法律B", "article_id": 1, "chunk_type": "parent"},
                "score": 0.7,
            }
        )

        self.assertNotEqual(first.evidence_id, second.evidence_id)
        self.assertNotEqual(first.law_name, second.law_name)
