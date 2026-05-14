"""意图路由器 - 纯关键词匹配（无BERT、无模型）"""

import re
from typing import Dict, Any


class IntentRouter:
    """
    意图路由器 - 纯关键词匹配

    路由顺序（优先级从高到低）：
    1. 法规关键词 → regulations（优先级最高，避免"招标法"被bids匹配）
    2. 价格关键词 → prices
    3. 招标关键词 → bids
    4. 默认 → bids（大部分问题是招标查询）
    """

    # 法规关键词（优先级最高）
    REGULATION_KEYWORDS = [
        "招标法", "投标法", "采购法", "政府采购法",
        "条例", "规定", "管理办法", "实施细则",
        "法规", "法律", "施行", "解读", "条款",
        "第.*条", "法律风险", "招标投标法律", "法律法规全书",
        "1200问", "评标方法", "定标", "废标", "流标"
    ]

    # 价格关键词
    PRICE_KEYWORDS = [
        "价格", "报价", "多少钱", "单价", "市场价", "行情"
    ]

    # 招标关键词
    BIDS_KEYWORDS = [
        "招标", "中标", "公告", "项目", "采购", "投标",
        "标书", "工程", "中标价", "中标金额", "中标人",
        "采购人", "代理机构", "预算", "招标文件", "投标文件"
    ]

    def __init__(self):
        self.regulation_pattern = re.compile(
            "|".join(self.REGULATION_KEYWORDS), re.IGNORECASE
        )
        self.price_pattern = re.compile(
            "|".join(self.PRICE_KEYWORDS), re.IGNORECASE
        )
        self.bids_pattern = re.compile(
            "|".join(self.BIDS_KEYWORDS), re.IGNORECASE
        )

    def route(self, query: str) -> Dict[str, Any]:
        """路由决策（按优先级顺序匹配）"""

        # 1. 优先检查法规关键词
        if self.regulation_pattern.search(query):
            return {
                "intent": "regulations",
                "collection": "regulations",
                "method": "hybrid"
            }

        # 2. 检查价格关键词
        if self.price_pattern.search(query):
            return {
                "intent": "prices",
                "collection": "prices",
                "method": "hybrid"
            }

        # 3. 检查招标关键词
        if self.bids_pattern.search(query):
            return {
                "intent": "bids",
                "collection": "bids",
                "method": "hybrid"
            }

        # 4. 默认去招标库（大部分问题是招标查询）
        return {
            "intent": "bids",
            "collection": "bids",
            "method": "hybrid"
        }
