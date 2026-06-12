# coding: utf-8

from agent_layer.adapters.base import BaseWebsiteAdapter
from agent_layer.schemas import Category


class ProductWebsiteAdapter(BaseWebsiteAdapter):
    expected_category = Category.PRODUCT
