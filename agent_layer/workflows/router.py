# coding: utf-8

from agent_layer.config import Settings
from agent_layer.schemas import Category, QuestionClassification, RouteDecision


class WorkflowRouter:
    def __init__(self, settings: Settings):
        self.settings = settings

    def decide(self, classification: QuestionClassification) -> RouteDecision:
        if classification.category == Category.OTHER:
            return RouteDecision(
                category=classification.category,
                workflow="general",
                action="general_answer",
                reason="这是一个通用问题，不需要查询招投标专业数据源，我会直接回答。",
            )

        if classification.confidence < self.settings.routing_confidence_low:
            return RouteDecision(
                category=classification.category,
                workflow=classification.category.value,
                action="clarify",
                reason="我还不能可靠判断您主要想查询哪类信息，需要先请您明确查询目标。",
            )

        if (
            classification.confidence < self.settings.routing_confidence_high
            and classification.secondary_categories
        ):
            return RouteDecision(
                category=classification.category,
                workflow=classification.category.value,
                action="clarify",
                reason="这个问题同时涉及多个业务方向，需要先确认您最关心的结论。",
            )

        return RouteDecision(
            category=classification.category,
            workflow=classification.category.value,
            action="execute",
            reason="问题类型已经比较明确，我会按对应的专业流程继续查询和核验。",
        )

    @staticmethod
    def clarification_message(classification: QuestionClassification) -> str:
        labels = {
            Category.POLICY: "政策法规依据",
            Category.TENDER: "招标项目或公告",
            Category.PUBLIC_OPINION: "舆情与风险事件",
            Category.COMPANY: "企业画像与能力",
            Category.PRICE: "价格与趋势",
            Category.PRODUCT: "商品参数与比较",
            Category.OTHER: "通用知识",
        }
        candidates = [classification.category, *classification.secondary_categories]
        unique = []
        for category in candidates:
            if category not in unique:
                unique.append(category)
        options = "、".join(labels[category] for category in unique[:3])
        return f"您的问题可能同时涉及{options}。请说明您最希望查询的结论或数据范围。"
