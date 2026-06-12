# coding: utf-8

from agent_layer.config import Settings
from agent_layer.schemas import Category
from agent_layer.workflows.common import DomainProfile


def build_tender_profile(settings: Settings) -> DomainProfile:
    return DomainProfile(
        category=Category.TENDER,
        display_name="招标信息",
        system_prompt="区分历史入库记录与网站实时公告，不得把历史结果表述为当前完整清单。",
        sql_views=["v_tender_project"],
        website_adapters=["tender_web"],
        required_evidence_fields=["project_id", "title", "region", "publish_date", "status"],
        freshness_policy="最新公告按小时级更新，回答必须说明数据截止时间和公告状态。",
        analysis_template="按地区、时间、类别和项目状态筛选；统计时说明样本量与筛选条件。",
        citation_policy="项目条件、中标结果和截止时间必须引用原始公告或可追溯记录。",
        local_collection=settings.tender_collection,
    )
