# coding: utf-8

from agent_layer.config import Settings
from agent_layer.schemas import Category
from agent_layer.workflows.common import DomainProfile


def build_public_opinion_profile(settings: Settings) -> DomainProfile:
    return DomainProfile(
        category=Category.PUBLIC_OPINION,
        display_name="舆情信息",
        system_prompt="区分独立事件与转载，关键负面结论需要权威来源或两个独立来源支撑。",
        sql_views=["v_public_opinion_event"],
        website_adapters=["public_opinion_web"],
        required_evidence_fields=["entity_id", "event_type", "published_at", "source_tier"],
        freshness_policy="舆情按分钟级更新，必须明确查询时间窗。",
        analysis_template="执行实体消歧、事件聚类、来源分级、情绪与趋势分析。",
        citation_policy="风险事实必须标注来源等级，不得以转载数量替代事件数量。",
    )
