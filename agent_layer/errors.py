# coding: utf-8


import httpx
from openai import APITimeoutError


class AgentError(Exception):
    code = "internal_error"
    user_message = "系统已记录错误编号。"


class DecompositionError(AgentError):
    code = "decomposition_failed"
    user_message = "问题拆解失败，无法可靠确定后续处理步骤。"


class PlanningError(AgentError):
    code = "planning_failed"
    user_message = "数据查询计划生成失败。"


class PolicyQueryError(AgentError):
    code = "policy_query_failed"
    user_message = "政策查询条件解析失败。"


class PolicyAssessmentError(AgentError):
    code = "policy_assessment_failed"
    user_message = "政策证据充分性判断失败。"


class GenerationError(AgentError):
    code = "generation_failed"
    user_message = "答案生成失败。"


class CitationValidationError(AgentError):
    code = "citation_validation_failed"
    user_message = "证据引用校验失败，系统未返回未经核验的结论。"


class PolicyRetrievalError(AgentError):
    code = "policy_retrieval_failed"
    user_message = "政策知识库检索失败，请检查知识库服务连接。"


class ModelTimeoutError(AgentError):
    code = "model_timeout"
    user_message = "模型调用超时。"


class WebsiteUnavailableError(AgentError):
    code = "website_unavailable"
    user_message = "网站数据源不可用，请检查网站适配器或网络连接。"


class SQLValidationError(AgentError):
    code = "sql_validation_failed"
    user_message = "结构化查询生成或校验失败。"


class SQLExecutionError(AgentError):
    code = "sql_execution_failed"
    user_message = "结构化数据库查询失败，请检查 SQL 服务连接或执行权限。"


class SQLTimeoutError(AgentError):
    code = "sql_timeout"
    user_message = "统计查询超时，请缩小查询范围。"


def raise_model_error(error: Exception, error_type: type[AgentError]) -> None:
    if isinstance(error, (TimeoutError, httpx.TimeoutException, APITimeoutError)):
        raise ModelTimeoutError from error
    raise error_type from error
