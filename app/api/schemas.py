"""请求/响应数据模型"""

from pydantic import BaseModel
from typing import List, Optional


class AskRequest(BaseModel):
    """问答请求"""
    question: str
    top_k: int = 5
    session_id: Optional[str] = None


class SourceInfo(BaseModel):
    """来源信息"""
    title: str = ""
    project_name: str = ""
    winner: str = ""
    #winner_amount: float = None
    winner_amount: Optional[float] = None
    content_preview: str = ""
    source_type: str = ""
    score: float = 0.0


class AskResponse(BaseModel):
    """问答响应"""
    answer: str
    sources: List[SourceInfo]
    processing_time: float
    session_id: str


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    collections: dict
    embedding_model: str
