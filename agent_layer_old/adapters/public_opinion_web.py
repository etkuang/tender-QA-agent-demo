# coding: utf-8

from agent_layer_old.adapters.base import BaseWebsiteAdapter
from agent_layer_old.schemas import Category


class PublicOpinionWebsiteAdapter(BaseWebsiteAdapter):
    expected_category = Category.PUBLIC_OPINION
