# coding: utf-8

import re


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

    def rewrite(self, question: str) -> str:
        result = question.strip()
        for source, target in self.punctuation_map.items():
            result = result.replace(source, target)
        return re.sub(r"\s+", " ", result)
