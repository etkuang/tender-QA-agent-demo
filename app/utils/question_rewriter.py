# app/utils/question_rewriter.py
"""
轻量级问题改写器 - 纯规则，0 API 调用
功能：口语转书面语、冗余精简、行业同义词替换
"""
import re
from typing import List, Tuple, Optional
from config import settings


class LightweightQuestionRewriter:
    """轻量级问题改写器 - 纯规则，不调用 LLM"""

    _instance: Optional['LightweightQuestionRewriter'] = None

    def __new__(cls) -> 'LightweightQuestionRewriter':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_rules()
        return cls._instance

    def _init_rules(self) -> None:
        """从配置加载规则"""
        # 开关
        self.enable_colloquial = getattr(settings, 'qr_enable_colloquial', True)
        self.enable_redundancy = getattr(settings, 'qr_enable_redundancy', True)
        self.enable_synonym = getattr(settings, 'qr_enable_synonym', True)

        # 口语映射（按长度降序，优先匹配长的）
        colloquial_mappings = getattr(settings, 'qr_colloquial_mappings', {})
        self.colloquial_mappings: List[Tuple[str, str]] = sorted(
            colloquial_mappings.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )

        # 冗余短语（按长度降序）
        redundant_phrases = getattr(settings, 'qr_redundant_phrases', [])
        self.redundant_phrases: List[str] = sorted(
            redundant_phrases,
            key=len,
            reverse=True
        )

        # 冗余正则模式
        self.redundancy_patterns: List[str] = getattr(settings, 'qr_redundancy_patterns', [])

        # 同义词映射（按长度降序）
        synonym_mappings = getattr(settings, 'qr_synonym_mappings', {})
        self.synonym_mappings: List[Tuple[str, str]] = sorted(
            synonym_mappings.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )

        # 标点规范化
        self.punctuation_map = {
            '？': '?',
            '！': '!',
            '；': ';',
            '：': ':',
            '，': ',',
            '。': '.',
            '、': ',',
            '“': '"',
            '”': '"',
            '‘': "'",
            '’': "'"
        }

    def rewrite(self, question: str) -> str:
        """
        执行改写（顺序：标点规范 → 冗余精简 → 口语转书面语 → 同义词替换）
        """
        if not question or not isinstance(question, str):
            return question or ""

        result = question

        # 0. 标点规范化（可选，始终执行）
        result = self._normalize_punctuation(result)

        # 1. 冗余精简（先做，减少后续处理量）
        if self.enable_redundancy:
            result = self._remove_redundancy(result)

        # 2. 口语转书面语
        if self.enable_colloquial:
            result = self._colloquial_to_formal(result)

        # 3. 同义词替换
        if self.enable_synonym:
            result = self._expand_synonyms(result)

        # 4. 清理多余空格和空问题
        result = re.sub(r'\s+', ' ', result).strip()

        # 5. 如果处理后为空，返回原问题
        if not result:
            return question

        if result != question:
            print(f"   ✍️ 规则改写: {question[:50]}... → {result[:50]}...")

        return result

    def _normalize_punctuation(self, text: str) -> str:
        """标点符号规范化"""
        result = text
        for cn, en in self.punctuation_map.items():
            result = result.replace(cn, en)
        return result

    def _remove_redundancy(self, text: str) -> str:
        """移除冗余"""
        result = text

        # 1. 精确移除冗余短语
        for phrase in self.redundant_phrases:
            while phrase in result:
                result = result.replace(phrase, "")

        # 2. 正则模式移除（如重复词）
        for pattern in self.redundancy_patterns:
            try:
                result = re.sub(pattern, "", result)
            except re.error:
                continue

        # 3. 移除连续重复的标点
        result = re.sub(r'([?！!。，,；;：:])\1+', r'\1', result)

        return result

    def _colloquial_to_formal(self, text: str) -> str:
        """口语转书面语"""
        result = text
        for colloquial, formal in self.colloquial_mappings:
            if colloquial in result:
                result = result.replace(colloquial, formal)
        return result

    def _expand_synonyms(self, text: str) -> str:
        """同义词替换"""
        result = text
        for oral, standard in self.synonym_mappings:
            if oral in result:
                result = result.replace(oral, standard)
        return result

    def is_enabled(self) -> bool:
        """检查是否有任何改写功能启用"""
        return self.enable_colloquial or self.enable_redundancy or self.enable_synonym


# 全局单例
question_rewriter = LightweightQuestionRewriter()