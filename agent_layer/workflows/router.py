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
                reason="问题被分类为通用问题，不调用专业数据工具。",
            )

        if classification.confidence < self.settings.routing_confidence_low:
            return RouteDecision(
                category=classification.category,
                workflow=classification.category.value,
                action="clarify",
                reason="分类置信度低，需要用户明确主要查询目标。",
            )

        if (
            classification.confidence < self.settings.routing_confidence_high
            and classification.secondary_categories
        ):
            return RouteDecision(
                category=classification.category,
                workflow=classification.category.value,
                action="clarify",
                reason="问题涉及多个接近的业务类别，需要确认主要诉求。",
            )

        return RouteDecision(
            category=classification.category,
            workflow=classification.category.value,
            action="execute",
            reason="分类结果达到执行阈值。",
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
