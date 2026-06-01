# coding: utf-8
# @Author: Wang Qingkang

from agent_layer.domain.retrieval.question_rewriter import question_rewriter
from agent_layer.infrastructure.config.agent_settings import settings


def rewrite_question(question: str) -> str:
    if not settings.enable_question_rewrite:
        return question
    if not question_rewriter.is_enabled():
        return question
    return question_rewriter.rewrite(question)