# app/utils/chinese_number.py
"""
中文数字转换工具 - 配置化版本
支持从配置文件读取映射表
"""
import re
from typing import Optional
from config import settings


class ChineseNumberConverter:
    """中文数字转换器（单例）"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_mapping()
        return cls._instance

    def _init_mapping(self):
        """从配置加载映射表"""
        self._mapping = settings.chinese_number_mapping
        self._enable_dynamic = settings.enable_dynamic_conversion
        self._enable_digits = settings.enable_digits
        self._enable_units = settings.enable_units

        # 中文数字字符
        self._digits = {'零': 0, '一': 1, '二': 2, '三': 3, '四': 4,
                        '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}

        # 中文单位
        self._units = {'十': 10, '百': 100, '千': 1000, '万': 10000}

    def to_arabic(self, chinese_num: str) -> str:
        """
        将中文数字转换为阿拉伯数字
        优先使用配置映射表，其次使用算法动态转换
        """
        if not chinese_num:
            return ""

        # 1. 优先从映射表查找
        if chinese_num in self._mapping:
            return self._mapping[chinese_num]

        # 2. 如果未启用动态转换，返回原值
        if not self._enable_dynamic:
            return chinese_num

        # 3. 动态算法转换
        return self._dynamic_convert(chinese_num)

    def _dynamic_convert(self, chinese_num: str) -> str:
        """动态转换中文数字到阿拉伯数字（支持1-9999）"""
        # 处理"十"开头的特殊情况（如"十五" -> 15）
        if chinese_num.startswith('十'):
            if len(chinese_num) == 1:
                return "10"
            else:
                result = 10
                for char in chinese_num[1:]:
                    if self._enable_digits and char in self._digits:
                        result += self._digits[char]
                return str(result)

        result = 0
        temp = 0

        for char in chinese_num:
            if self._enable_digits and char in self._digits:
                temp = self._digits[char]
            elif self._enable_units and char in self._units:
                unit = self._units[char]
                if temp == 0:
                    temp = 1
                result += temp * unit
                temp = 0
            else:
                return chinese_num

        result += temp
        return str(result)

    def extract_article_number(self, text: str) -> str:
        """从文本中提取条款号（支持阿拉伯数字和中文数字）"""
        # 匹配阿拉伯数字
        match = re.search(r'第\s*(\d+)\s*条', text)
        if match:
            return match.group(1)

        # 匹配中文数字
        match = re.search(r'第([一二三四五六七八九十百千万零]+)条', text)
        if match:
            chinese = match.group(1)
            return self.to_arabic(chinese)

        return ""


# 全局单例
chinese_number_converter = ChineseNumberConverter()


# 便捷函数（保持向后兼容）
def chinese_to_arabic_dynamic(chinese_num: str) -> str:
    """动态转换中文数字到阿拉伯数字（兼容旧接口）"""
    return chinese_number_converter.to_arabic(chinese_num)


def chinese_to_arabic(chinese_num: str) -> str:
    """中文数字转阿拉伯数字（兼容旧接口）"""
    return chinese_number_converter.to_arabic(chinese_num)


def extract_article_number(text: str) -> str:
    """从文本中提取条款号（兼容旧接口）"""
    return chinese_number_converter.extract_article_number(text)