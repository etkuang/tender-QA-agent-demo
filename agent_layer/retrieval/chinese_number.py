# coding: utf-8

import re


class ChineseNumberConverter:
    digits = {
        "零": 0,
        "〇": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    units = {"十": 10, "百": 100, "千": 1000}

    def to_arabic(self, value: str) -> str:
        if value.isdigit():
            return value

        total = 0
        current = 0
        for char in value:
            if char in self.digits:
                current = self.digits[char]
                continue
            if char in self.units:
                unit = self.units[char]
                total += (current if current else 1) * unit
                current = 0

        total += current
        if total > 0:
            return f"{total}"
        return value

    def extract_article_number(self, text: str) -> str:
        digit_match = re.search(r"第\s*(\d+)\s*条", text)
        if digit_match:
            return digit_match.group(1)

        chinese_match = re.search(r"第\s*([零〇一二两三四五六七八九十百千]+)\s*条", text)
        if chinese_match:
            return self.to_arabic(chinese_match.group(1))

        loose_match = re.search(r"\d+", text)
        if loose_match:
            return loose_match.group(0)
        return ""
