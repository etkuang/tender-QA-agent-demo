# coding: utf-8
# @Author: Wang Qingkang

from agent_layer.adapters.base import WebsiteSearchClient
from agent_layer.schemas import Category


def normalize_website_clients(
    website_clients: dict[str, WebsiteSearchClient] | None,
    policy_internet_client: WebsiteSearchClient | None,
    general_internet_client: WebsiteSearchClient | None,
) -> dict[str, WebsiteSearchClient]:
    clients = dict(website_clients or {})
    if policy_internet_client is not None:
        clients[Category.POLICY.value] = policy_internet_client
    if general_internet_client is not None:
        clients[Category.OTHER.value] = general_internet_client
    legacy_names = {
        "tender_web": Category.TENDER.value,
        "public_opinion_web": Category.PUBLIC_OPINION.value,
        "company_web": Category.COMPANY.value,
        "product_web": Category.PRODUCT.value,
    }
    for old_name, category_name in legacy_names.items():
        if old_name in clients and category_name not in clients:
            clients[category_name] = clients[old_name]
    return clients