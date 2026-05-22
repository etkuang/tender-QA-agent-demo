# app/core/config_loader.py
import yaml
import os
from pathlib import Path
from typing import Dict, List, Any, Optional


class ConfigLoader:
    _instance = None
    _config: Dict = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self, config_path: str = "config.yaml"):
        """加载配置文件"""
        if not self._config:
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    self._config = yaml.safe_load(f)
                print(f"✅ 加载配置文件: {config_path}")
            else:
                print(f"⚠️ 配置文件不存在: {config_path}，使用默认值")
                self._config = self._get_default_config()
        return self._config

    def _get_default_config(self) -> Dict:
        """默认配置（硬编码后备）"""
        return {
            "retrieval": {
                "weights": {"keyword_heavy": [0.75, 0.25], "semantic_heavy": [0.40, 0.60], "balanced": [0.65, 0.35]},
                "boost_keywords": {"high": [], "medium": [], "low": []},
                "regulation_keywords": {"high": [], "medium": []},
                "penalty_patterns": []
            },
            "intent": {
                "greeting_responses": {},
                "greeting_keywords": [],
                "thanks_keywords": [],
                "goodbye_keywords": [],
                "unrelated_keywords": [],
                "bidding_keywords": [],
                "keyword_heavy_patterns": [],
                "semantic_heavy_patterns": []
            },
            "react_agent": {"max_steps": 5, "temperature": 0.3, "max_tokens": 1000},
            "retriever": {"top_k": 5, "vector_recall": 20, "summarize_max_chunks": 8},
            "server": {"host": "0.0.0.0", "port": 8000, "debug": True},
            "pdf": {"pdf_dir": "./data/pdfs", "chroma_persist_dir": "./data/databases/chroma_db"}
        }

    def get(self, key: str, default=None):
        """获取配置值，支持点号分隔的路径"""
        keys = key.split('.')
        value = self._config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default


    def get_app_config(self) -> dict:
        """获取应用配置"""
        return self._config.get("app", {})


    def get_response_messages(self) -> dict:
        """获取响应消息配置"""
        return self._config.get("responses", {})

# 全局实例
config_loader = ConfigLoader()