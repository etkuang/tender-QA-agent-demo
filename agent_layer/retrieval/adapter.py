# coding: utf-8
# @Author: Wang Qingkang

from langchain_core.documents import Document

from agent_layer.config import settings
from agent_layer.retrieval.retriever import HybridRetriever


class LegalRetrievalAdapter:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def retrieve(self, question: str) -> list[Document]:
        results = self.retriever.search(question, settings.default_collection, top_k=settings.top_k)
        documents = []
        for item in results:
            metadata = {
                **item.get("data", {}),
                "score": item.get("score", 0),
                "id": item.get("id", ""),
            }
            documents.append(Document(page_content=item.get("text", ""), metadata=metadata))
        return documents

    def format_documents(self, documents: list[Document]) -> str:
        blocks = []
        for index, document in enumerate(documents[: settings.summarize_max_chunks], 1):
            text = document.page_content[: settings.summarize_chunk_length]
            blocks.append(f"[{index}] 来源：{self._source_label(document)}\n{text}")
        return "\n\n".join(blocks)

    def build_sources(self, documents: list[Document]) -> list[dict]:
        output = []
        for document in documents[: settings.top_k]:
            output.append(
                {
                    "title": self._source_label(document),
                    "content_preview": document.page_content[:200],
                    "score": document.metadata.get("score", 0),
                }
            )
        return output

    @staticmethod
    def _source_label(document: Document) -> str:
        metadata = document.metadata
        title = metadata.get("doc_title") or metadata.get("law_name") or metadata.get("source") or "未知来源"
        article_num = metadata.get("article_num") or metadata.get("article_id")
        if article_num and article_num not in ("unknown", "full"):
            return f"{title} 第{article_num}条"
        return title
