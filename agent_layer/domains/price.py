# coding: utf-8

from agent_layer.config import Settings
from agent_layer.schemas import Category
from agent_layer.workflows.common import DomainProfile


def build_price_profile(settings: Settings) -> DomainProfile:
    return DomainProfile(
        category=Category.PRICE,
        display_name="价格信息",
        system_prompt="统一币种、税口径、单位、规格、地区和时间，不做脱离规格条件的绝对比较。",
        sql_views=["v_price_history"],
        website_adapters=["price_web"],
        required_evidence_fields=["product_id", "specification", "unit", "region", "tax_basis", "price"],
        freshness_policy="报价时效按来源配置，回答必须显示报价或成交日期。",
        analysis_template="输出样本量、时间范围、分位数、趋势和异常值处理规则。",
        citation_policy="每个关键价格数字必须可映射到查询结果或原始报价来源。",
        local_collection=settings.price_collection,
    )
