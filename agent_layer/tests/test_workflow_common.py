# coding: utf-8

import unittest

from agent_layer.workflows.common import citations_are_valid


class WorkflowCommonTests(unittest.TestCase):
    def test_citation_markers_must_exist_and_stay_in_range(self):
        self.assertTrue(citations_are_valid("结论。[1] 补充。[2]", 2))
        self.assertFalse(citations_are_valid("没有引用。", 2))
        self.assertFalse(citations_are_valid("越界引用。[3]", 2))
        self.assertTrue(citations_are_valid("无证据回答。", 0))


if __name__ == "__main__":
    unittest.main()
