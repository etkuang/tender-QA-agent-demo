# coding: utf-8

from agent_layer_old.adapters.base import BaseWebsiteAdapter
from agent_layer_old.schemas import Category


class PolicyInternetAdapter(BaseWebsiteAdapter):
    expected_category = Category.POLICY
