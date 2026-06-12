# coding: utf-8

import unittest

from langchain_core.runnables import RunnableLambda

from agent_layer.config import Settings
from agent_layer.context import ContextResolver
from agent_layer.retrieval.chinese_number import ChineseNumberConverter
from agent_layer.retrieval.rewrite import QuestionRewriter
from agent_layer.schemas import ContextResolution, Message, SessionContext


class FakeStructuredModel:
    def with_structured_output(self, schema, **kwargs):
        return RunnableLambda(
            lambda _: ContextResolution(
                standalone_question="A公司有没有处罚记录？",
                ambiguous=False,
            )
        )


class ContextResolverTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        settings = Settings()
        self.resolver = ContextResolver(QuestionRewriter(), FakeStructuredModel(), settings)

    async def test_reference_question_keeps_previous_user_topic(self):
        history = [Message(role="user", content="A公司近三年有哪些中标项目？")]

        result = await self.resolver.resolve("该公司有没有处罚记录？", history, SessionContext())

        self.assertEqual(result.standalone_question, "A公司有没有处罚记录？")

    async def test_year_is_not_misread_as_article_number(self):
        converter = ChineseNumberConverter()

        result = converter.extract_article_number("查询2023年上海的中标项目")

        self.assertEqual(result, "")
