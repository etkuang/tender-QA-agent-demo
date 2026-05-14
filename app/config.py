"""配置管理"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用配置"""

    # LLM配置
    llm_api_key: str = ""
    llm_api_url: str = ""
    llm_model: str = "spark-v3.5"

    # 数据路径
    data_path_bids: str = "./data/招标采购标的物信息提取训练数据.xlsx"
    data_path_prices: str = "./data/price_data.xlsx"
    pdf_dir: str = "./data/pdfs"

    # Chroma配置
    chroma_persist_dir: str = "./chroma_db"

    # Redis配置
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    # 检索配置
    top_k: int = 5
    vector_recall: int = 50
    embedding_model: str = "BAAI/bge-small-zh"
    use_reranker: bool = True

    # 会话配置
    session_ttl: int = 1800
    max_history: int = 10

    # 服务配置
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
