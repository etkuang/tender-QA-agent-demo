# coding: utf-8
# @Author: Wang Qingkang

from agent_layer.schemas import Category
from agent_layer.data_domain.schemas import (
    ApprovedSQLExample,
    BusinessGlossaryTerm,
    DataColumnContext,
    DataMetricDefinition,
    DataRelationshipContext,
    DataTableContext,
)


class DataContextCatalog:
    def __init__(
        self,
        tables: list[DataTableContext],
        relationships: list[DataRelationshipContext],
        metrics: list[DataMetricDefinition],
        glossary_terms: list[BusinessGlossaryTerm],
        examples: list[ApprovedSQLExample],
    ):
        self.tables = {table.name: table for table in tables}
        self.relationships = relationships
        self.metrics = metrics
        self.glossary_terms = glossary_terms
        self.examples = examples

    def tables_for_domain(self, category: Category) -> list[DataTableContext]:
        return [
            table
            for table in self.tables.values()
            if table.domain == category
        ]

    def relationships_for_tables(self, table_names: set[str]) -> list[DataRelationshipContext]:
        return [
            relationship
            for relationship in self.relationships
            if relationship.approved
            and relationship.left_table in table_names
            and relationship.right_table in table_names
        ]

    def metrics_for_domain(self, category: Category) -> list[DataMetricDefinition]:
        return [
            metric
            for metric in self.metrics
            if metric.domain == category
        ]

    def glossary_for_domain(self, category: Category) -> list[BusinessGlossaryTerm]:
        return [
            term
            for term in self.glossary_terms
            if term.domain == category
        ]

    def examples_for_domain(self, category: Category) -> list[ApprovedSQLExample]:
        return [
            example
            for example in self.examples
            if example.domain == category
        ]


def build_default_data_context_catalog() -> DataContextCatalog:
    tables = [
        DataTableContext(
            name="v_tender_project",
            domain=Category.TENDER,
            description="招标项目、公告和中标结果分析视图。每行表示一个已入库项目或公告记录。",
            primary_key=["project_id"],
            default_time_column="publish_date",
            default_entity_column="project_id",
            freshness="历史入库记录，不代表实时完整公告清单。",
            columns=[
                DataColumnContext(name="project_id", description="稳定项目 ID", data_type="text", semantic_type="identifier"),
                DataColumnContext(name="title", description="公告或项目标题", data_type="text"),
                DataColumnContext(name="category", description="采购类别", data_type="text"),
                DataColumnContext(name="region", description="行政区域", data_type="text"),
                DataColumnContext(name="publish_date", description="公告发布日期", data_type="date"),
                DataColumnContext(name="budget", description="预算金额，按视图约定币种和单位", data_type="numeric"),
                DataColumnContext(name="winner", description="中标人名称", data_type="text"),
                DataColumnContext(name="amount", description="中标金额，按视图约定币种和单位", data_type="numeric"),
            ],
        ),
        DataTableContext(
            name="v_company_profile",
            domain=Category.COMPANY,
            description="企业主体和历史中标画像视图。每行表示一个稳定企业主体。",
            primary_key=["company_id"],
            default_time_column="latest_bid_date",
            default_entity_column="company_id",
            freshness="企业基础信息按天更新，处罚和经营异常需要更短时效来源补充。",
            columns=[
                DataColumnContext(name="company_id", description="稳定企业 ID", data_type="text", semantic_type="identifier"),
                DataColumnContext(name="credit_code", description="统一社会信用代码", data_type="text", semantic_type="identifier", is_sensitive=True),
                DataColumnContext(name="bid_count", description="历史中标次数", data_type="integer"),
                DataColumnContext(name="total_amount", description="历史中标总金额，按视图约定币种和单位", data_type="numeric"),
                DataColumnContext(name="categories", description="历史覆盖采购类别", data_type="text"),
                DataColumnContext(name="regions", description="历史覆盖地区", data_type="text"),
                DataColumnContext(name="latest_bid_date", description="最近中标日期", data_type="date"),
            ],
        ),
        DataTableContext(
            name="v_product_catalog",
            domain=Category.PRODUCT,
            description="商品目录与标准化参数视图。每行表示一个标准化商品型号。",
            primary_key=["product_id"],
            default_time_column="updated_at",
            default_entity_column="product_id",
            freshness="商品参数按厂商版本和页面更新时间判断有效性。",
            columns=[
                DataColumnContext(name="product_id", description="稳定商品 ID", data_type="text", semantic_type="identifier"),
                DataColumnContext(name="brand", description="品牌", data_type="text"),
                DataColumnContext(name="model", description="型号", data_type="text"),
                DataColumnContext(name="category", description="品类", data_type="text"),
                DataColumnContext(name="parameter_json", description="标准化参数 JSON", data_type="json"),
                DataColumnContext(name="source", description="数据来源", data_type="text"),
                DataColumnContext(name="updated_at", description="更新时间", data_type="datetime"),
            ],
        ),
        DataTableContext(
            name="v_public_opinion_event",
            domain=Category.PUBLIC_OPINION,
            description="已聚类的企业或项目舆情事件视图。每行表示一个舆情事件聚类。",
            primary_key=["event_cluster_id"],
            default_time_column="published_at",
            default_entity_column="entity_id",
            freshness="舆情按分钟级更新，必须明确查询时间窗。",
            columns=[
                DataColumnContext(name="entity_id", description="稳定主体 ID", data_type="text", semantic_type="identifier"),
                DataColumnContext(name="event_type", description="事件类型", data_type="text"),
                DataColumnContext(name="sentiment", description="情绪方向", data_type="text"),
                DataColumnContext(name="source", description="来源", data_type="text"),
                DataColumnContext(name="published_at", description="发布时间", data_type="datetime"),
                DataColumnContext(name="event_cluster_id", description="独立事件聚类 ID", data_type="text", semantic_type="identifier"),
            ],
        ),
    ]
    relationships = [
        DataRelationshipContext(
            left_table="v_tender_project",
            left_columns=["winner"],
            right_table="v_company_profile",
            right_columns=["company_id"],
            relationship_type="many_to_one",
            description="部署侧应优先用稳定企业 ID 连接；仅有名称时必须说明可能存在重名歧义。",
        ),
    ]
    metrics = [
        DataMetricDefinition(
            name="招标项目数量",
            domain=Category.TENDER,
            description="满足筛选条件的项目或公告记录数。",
            formula="COUNT(DISTINCT project_id)",
            grain="project_id",
            required_tables=["v_tender_project"],
            required_columns=["project_id"],
            time_logic="默认按 publish_date 过滤。",
            caveats=["历史入库范围不等于全网实时公告范围。"],
        ),
        DataMetricDefinition(
            name="中标金额",
            domain=Category.TENDER,
            description="满足筛选条件的中标金额合计。",
            formula="SUM(amount)",
            grain="project_id",
            required_tables=["v_tender_project"],
            required_columns=["amount"],
            filters=["amount IS NOT NULL"],
            time_logic="默认按 publish_date 过滤。",
            caveats=["币种和单位以视图约定为准。"],
        ),
        DataMetricDefinition(
            name="历史中标次数",
            domain=Category.COMPANY,
            description="企业画像中记录的历史中标次数。",
            formula="SUM(bid_count)",
            grain="company_id",
            required_tables=["v_company_profile"],
            required_columns=["bid_count"],
            caveats=["企业名称无法稳定消歧时不能直接合并统计。"],
        ),
        DataMetricDefinition(
            name="负面舆情事件数",
            domain=Category.PUBLIC_OPINION,
            description="情绪方向为负面的独立事件聚类数量。",
            formula="COUNT(DISTINCT event_cluster_id)",
            grain="event_cluster_id",
            required_tables=["v_public_opinion_event"],
            required_columns=["event_cluster_id", "sentiment"],
            filters=["sentiment = 'negative'"],
            time_logic="默认按 published_at 过滤。",
            caveats=["转载数量不能替代独立事件数量。"],
        ),
    ]
    glossary_terms = [
        BusinessGlossaryTerm(
            term="最新",
            domain=Category.TENDER,
            definition="默认指按公告发布日期 publish_date 倒序的最近记录；如果用户要求实时最新，需要网站补充。",
            synonyms=["最近", "近期"],
            ambiguity_notes=["需要确认时间窗口时应澄清。"],
        ),
        BusinessGlossaryTerm(
            term="收入",
            domain=Category.TENDER,
            definition="招投标上下文中通常不直接等同于中标金额；如用户询问收入，应澄清指标口径。",
            ambiguity_notes=["可能指预算、中标金额、合同金额或企业财务收入。"],
        ),
        BusinessGlossaryTerm(
            term="负面舆情",
            domain=Category.PUBLIC_OPINION,
            definition="sentiment 为 negative 的独立事件聚类，不是转载文章条数。",
        ),
    ]
    examples = [
        ApprovedSQLExample(
            name="按地区统计项目数量",
            domain=Category.TENDER,
            question="统计上个月各地区招标项目数量",
            sql="SELECT region, COUNT(DISTINCT project_id) AS project_count FROM v_tender_project WHERE publish_date >= :start_date AND publish_date < :end_date GROUP BY region ORDER BY project_count DESC LIMIT 200",
            tables=["v_tender_project"],
        ),
        ApprovedSQLExample(
            name="企业画像查询",
            domain=Category.COMPANY,
            question="查询某企业历史中标情况",
            sql="SELECT company_id, bid_count, total_amount, categories, regions, latest_bid_date FROM v_company_profile WHERE company_id = :company_id LIMIT 20",
            tables=["v_company_profile"],
        ),
    ]
    return DataContextCatalog(tables, relationships, metrics, glossary_terms, examples)