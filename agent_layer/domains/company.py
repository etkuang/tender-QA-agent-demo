# coding: utf-8

from agent_layer.config import Settings
from agent_layer.schemas import Category
from agent_layer.workflows.common import DomainProfile


def build_company_profile(settings: Settings) -> DomainProfile:
    return DomainProfile(
        category=Category.COMPANY,
        display_name="企业信息",
        system_prompt="优先使用统一社会信用代码或稳定企业 ID 消歧，名称无法确认时必须说明歧义。",
        sql_views=["v_company_profile"],
        website_adapters=["company_web"],
        required_evidence_fields=["company_id", "credit_code", "data_as_of"],
        freshness_policy="企业基本信息按天更新，处罚和经营异常采用更短时效。",
        analysis_template="结合历史中标次数、金额、地区和类别覆盖分析能力与风险。",
        citation_policy="回答说明企业标识、统计口径、截止时间和未知项。",
        knowledge_index=None,
    )
