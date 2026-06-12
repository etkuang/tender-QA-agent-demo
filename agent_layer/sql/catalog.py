# coding: utf-8

from agent_layer.sql.schemas import ViewSchema


def build_default_schema_catalog() -> dict[str, ViewSchema]:
    schemas = [
        ViewSchema(
            name="v_tender_project",
            description="招标项目、公告和中标结果分析视图。",
            columns={
                "project_id": "稳定项目 ID",
                "title": "公告或项目标题",
                "category": "采购类别",
                "region": "行政区域",
                "publish_date": "公告发布日期",
                "budget": "预算金额及约定币种单位",
                "winner": "中标人名称",
                "amount": "中标金额及约定币种单位",
            },
        ),
        ViewSchema(
            name="v_company_profile",
            description="企业主体和历史中标画像视图。",
            columns={
                "company_id": "稳定企业 ID",
                "credit_code": "统一社会信用代码",
                "bid_count": "中标次数",
                "total_amount": "中标总金额",
                "categories": "覆盖类别",
                "regions": "覆盖地区",
                "latest_bid_date": "最近中标日期",
            },
        ),
        ViewSchema(
            name="v_price_history",
            description="同规格商品或服务的历史成交价格视图。",
            columns={
                "product_id": "稳定商品 ID",
                "specification": "规格型号",
                "unit": "计量单位",
                "region": "成交地区",
                "tax_basis": "含税口径",
                "price": "成交价格",
                "transaction_date": "成交日期",
            },
        ),
        ViewSchema(
            name="v_product_catalog",
            description="商品目录与标准化参数视图。",
            columns={
                "product_id": "稳定商品 ID",
                "brand": "品牌",
                "model": "型号",
                "category": "品类",
                "parameter_json": "标准化参数",
                "source": "数据来源",
                "updated_at": "更新时间",
            },
        ),
        ViewSchema(
            name="v_public_opinion_event",
            description="已聚类的企业或项目舆情事件视图。",
            columns={
                "entity_id": "稳定主体 ID",
                "event_type": "事件类型",
                "sentiment": "情绪方向",
                "source": "来源",
                "published_at": "发布时间",
                "event_cluster_id": "独立事件聚类 ID",
            },
        ),
    ]
    return {schema.name: schema for schema in schemas}
