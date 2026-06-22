# coding: utf-8


import httpx
from openai import APITimeoutError


class AgentError(Exception):
    code = "internal_error"
    user_message = "系统已记录错误编号。"


class ClassificationError(AgentError):
    code = "classification_failed"
    user_message = "暂时无法识别问题类型，请稍后重试。"


class PlanningError(AgentError):
    code = "internal_error"
    user_message = "暂时无法生成可靠的研究计划，请稍后重试。"


class PolicyQueryError(AgentError):
    code = "internal_error"
    user_message = "暂时无法解析政策查询条件。"


class PolicyAssessmentError(AgentError):
    code = "internal_error"
    user_message = "暂时无法判断政策证据是否充分。"


class GenerationError(AgentError):
    code = "internal_error"
    user_message = "暂时无法基于证据生成回答。"


class CitationValidationError(AgentError):
    code = "internal_error"
    user_message = "证据引用校验未通过，系统未返回未经核验的结论。"


class PolicyRetrievalError(AgentError):
    code = "policy_retrieval_failed"
    user_message = "政策知识库暂时不可用。"


class ModelTimeoutError(AgentError):
    code = "model_timeout"
    user_message = "回答生成超时，请稍后重试。"


class WebsiteUnavailableError(AgentError):
    code = "website_unavailable"
    user_message = "部分实时来源暂时不可用，已基于可用资料回答。"


class SQLValidationError(AgentError):
    code = "sql_validation_failed"
    user_message = "当前统计请求无法安全执行。"


class SQLTimeoutError(AgentError):
    code = "sql_timeout"
    user_message = "统计查询超时，请缩小范围。"


def raise_model_error(error: Exception, error_type: type[AgentError]) -> None:
    if isinstance(error, (httpx.TimeoutException, APITimeoutError)):
        raise ModelTimeoutError from error
    raise error_type from error
