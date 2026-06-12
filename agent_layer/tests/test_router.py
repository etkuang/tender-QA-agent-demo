# coding: utf-8

import unittest

from agent_layer.config import Settings
from agent_layer.schemas import Category, QuestionClassification
from agent_layer.workflows.router import WorkflowRouter


class WorkflowRouterTests(unittest.TestCase):
    def setUp(self):
        self.router = WorkflowRouter(Settings())

    def test_other_uses_general_answer(self):
        classification = QuestionClassification(
            category=Category.OTHER,
            confidence=0.99,
            reasoning="通用知识问题",
        )

        decision = self.router.decide(classification)

        self.assertEqual(decision.action, "general_answer")
        self.assertEqual(decision.workflow, "general")

    def test_low_confidence_requests_clarification(self):
        classification = QuestionClassification(
            category=Category.COMPANY,
            confidence=0.30,
            reasoning="主体和诉求不明确",
        )

        decision = self.router.decide(classification)

        self.assertEqual(decision.action, "clarify")

    def test_high_confidence_executes_domain_workflow(self):
        classification = QuestionClassification(
            category=Category.POLICY,
            confidence=0.93,
            reasoning="询问处罚依据",
        )

        decision = self.router.decide(classification)

        self.assertEqual(decision.action, "execute")
        self.assertEqual(decision.workflow, "policy")
