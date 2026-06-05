# coding: utf-8

import re

from agent_layer.config import settings


class QuestionRewriter:
    punctuation_map = {
        "？": "?",
        "！": "!",
        "；": ";",
        "：": ":",
        "，": ",",
        "。": ".",
        "、": ",",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
    }

    def __init__(self):
        self.colloquial_mappings = sorted(
            settings.colloquial_mappings.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        )
        self.redundant_phrases = sorted(settings.redundant_phrases, key=len, reverse=True)
        self.synonym_mappings = sorted(
            settings.synonym_mappings.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        )

    def rewrite(self, question: str) -> str:
        if not settings.enable_question_rewrite:
            return question
        if not question:
            return ""

        result = question
        for source, target in self.punctuation_map.items():
            result = result.replace(source, target)
        for phrase in self.redundant_phrases:
            result = result.replace(phrase, "")
        for pattern in settings.redundancy_patterns:
            result = re.sub(pattern, "", result)
        for source, target in self.colloquial_mappings:
            result = result.replace(source, target)
        for source, target in self.synonym_mappings:
            result = result.replace(source, target)

        result = re.sub(r"\s+", " ", result).strip()
        return result if result else question
