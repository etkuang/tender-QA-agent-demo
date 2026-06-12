# coding: utf-8

import unittest

from agent_layer.app import ApplicationDependencies, TenderQAApplication
from agent_layer.config import Settings
from agent_layer.schemas import (
    Category,
    ConversationInput,
    Evidence,
    QuestionClassification,
    SourceType,
    StreamEventType,
    WorkflowResult,
)
from agent_layer.workflows.router import WorkflowRouter


class FakeClassifier:
    async def classify(self, question: str, context_summary: str = "") -> QuestionClassification:
        return QuestionClassification(
            category=Category.POLICY,
            confidence=0.95,
            reasoning="询问法规处罚",
        )


class FakePolicyWorkflow:
    async def run(self, original_question: str, standalone_question: str, runtime_context) -> WorkflowResult:
        evidence = Evidence(
            evidence_id="evidence-1",
            domain=Category.POLICY,
            source_type=SourceType.LOCAL_DOCUMENT,
            title="招标投标法 第五十三条",
            content="串通投标处罚规定",
        )
        return WorkflowResult(answer="串通投标应依法处罚。[1]", evidence=[evidence])


class FakeGeneralWorkflow:
    async def run(self, question: str, history: str) -> str:
        return "通用回答"


class FakeContextResolver:
    async def resolve(self, question: str, history_messages: list, session_context) -> ConversationInput:
        return ConversationInput(
            original_question=question,
            standalone_question=question,
            history_messages=history_messages,
            session_context=session_context,
        )

    def context_summary(self, conversation: ConversationInput) -> str:
        return ""


class TenderQAApplicationTests(unittest.IsolatedAsyncioTestCase):
    async def test_policy_route_streams_target_events(self):
        settings = Settings(stream_chunk_size=4)
        dependencies = ApplicationDependencies(
            settings=settings,
            context_resolver=FakeContextResolver(),
            classifier=FakeClassifier(),
            router=WorkflowRouter(settings),
            general_workflow=FakeGeneralWorkflow(),
            policy_workflow=FakePolicyWorkflow(),
            data_workflows={},
        )
        application = TenderQAApplication(dependencies)

        events = [event async for event in application.stream("串通投标如何处罚？")]

        answer = "".join(event.content for event in events if event.type == StreamEventType.ASSISTANT_DELTA)
        self.assertEqual(answer, "串通投标应依法处罚。[1]")
        self.assertTrue(any(event.type == StreamEventType.ROUTE for event in events))
        self.assertTrue(any(event.type == StreamEventType.REASONING_SUMMARY for event in events))
        self.assertTrue(any(event.type == StreamEventType.SOURCE for event in events))
        self.assertEqual(events[-1].type, StreamEventType.FINAL)

    async def test_quick_response_does_not_call_classifier(self):
        settings = Settings()
        dependencies = ApplicationDependencies(
            settings=settings,
            context_resolver=FakeContextResolver(),
            classifier=FakeClassifier(),
            router=WorkflowRouter(settings),
            general_workflow=FakeGeneralWorkflow(),
            policy_workflow=FakePolicyWorkflow(),
            data_workflows={},
        )
        application = TenderQAApplication(dependencies)

        result = await application.ask("你好")

        self.assertEqual(result.route, "greeting")
        self.assertIn("您好", result.answer)
