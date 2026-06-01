# coding: utf-8
# @Author: Wang Qingkang

import re

from agent_layer.infrastructure.config.agent_settings import settings


def select_fusion_method(query: str) -> str:
    if settings.fusion_strategy != "smart":
        return settings.fusion_strategy

    if re.search(r"什么是|定义|解释|含义|如何|怎么|步骤|流程|多少|交|办", query):
        return "weighted"
    if re.search(r"区别|不同|对比|第\\d+条", query):
        return "rrf"
    if re.search(r"处罚|罚款|责任", query):
        return "weighted"
    if len(query) > 15:
        return "weighted"
    return "weighted"