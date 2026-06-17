# coding: utf-8

from agent_layer.config import Settings
from agent_layer.schemas import Category
from agent_layer.workflows.common import DomainProfile


def build_product_profile(settings: Settings) -> DomainProfile:
    return DomainProfile(
        category=Category.PRODUCT,
        display_name="商品信息",
        system_prompt="先归一品牌、型号、品类和参数名；缺失参数标为未知，不得猜测。",
        sql_views=["v_product_catalog"],
        website_adapters=["product_web"],
        required_evidence_fields=["product_id", "brand", "model", "parameter_json", "updated_at"],
        freshness_policy="商品参数按厂商版本和页面更新时间判断有效性。",
        analysis_template="按相同参数维度对齐比较，并解释用户需求与商品参数的对应关系。",
        citation_policy="参数和适配结论必须标明商品主数据或厂商页面来源。",
    )
