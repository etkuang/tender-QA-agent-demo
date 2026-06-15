# coding: utf-8
# @Author: Wang Qingkang


import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Literal

import pymupdf
from pydantic import BaseModel

from knowledge_base_layer.config import Settings
from knowledge_base_layer.ingestion.normalizer import ChunkNormalizer, NormalizedChunk


ARTICLE_PATTERN = re.compile(r"^第\s*([零〇一二两三四五六七八九十百千0-9]+)\s*条")
LAW_TITLE_PATTERN = re.compile(r"^[\u4e00-\u9fff·]+(?:法|条例|办法|规定|规则)$")


class PdfSource(BaseModel):
    path: str
    title: str
    source_kind: Literal["legal_compendium", "commentary"]
    layout: Literal["single_column", "two_column"]
    source_as_of: str
    authority_level: int
    freshness_level: int
    first_page: int
    last_page: int | None = None


class PdfPage(BaseModel):
    page_number: int
    text: str


def load_pdf_sources(manifest_path: Path) -> list[PdfSource]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [PdfSource.model_validate(item) for item in payload]


class PolicyPdfLoader:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.normalizer = ChunkNormalizer()

    def load(self, source: PdfSource, source_root: Path, data_version: str) -> list[NormalizedChunk]:
        pages = self._extract_pages(source_root / source.path, source)
        if source.source_kind == "legal_compendium":
            return self._legal_chunks(pages, source, data_version)
        return self._commentary_chunks(pages, source, data_version)

    def _extract_pages(self, path: Path, source: PdfSource) -> list[PdfPage]:
        document = pymupdf.open(path)
        last_page = source.last_page or document.page_count
        pages = []
        for page_index in range(source.first_page - 1, last_page):
            page = document[page_index]
            blocks = [block for block in page.get_text("blocks") if block[4].strip()]
            if source.layout == "two_column":
                midpoint = page.rect.width / 2
                left = sorted((block for block in blocks if block[0] < midpoint), key=lambda item: (item[1], item[0]))
                right = sorted((block for block in blocks if block[0] >= midpoint), key=lambda item: (item[1], item[0]))
                blocks = left + right
            else:
                blocks = sorted(blocks, key=lambda item: (item[1], item[0]))
            text = "\n".join(block[4] for block in blocks)
            pages.append(PdfPage(page_number=page_index + 1, text=self._normalize_text(text)))
        document.close()
        return pages

    def _legal_chunks(
        self,
        pages: list[PdfPage],
        source: PdfSource,
        data_version: str,
    ) -> list[NormalizedChunk]:
        chunks = []
        law_name = ""
        article_id = ""
        article_page = 0
        article_lines = []

        def flush_article() -> None:
            if not law_name or not article_id or not article_lines:
                return
            content = "\n".join(article_lines).strip()
            parent_id = self._stable_id(source.path, law_name, article_id, content)
            metadata = self._metadata(source, article_page)
            metadata.update(
                {
                    "law_name": law_name,
                    "article_id": article_id,
                    "document_id": parent_id,
                    "chunk_type": "parent",
                    "validity_status": "source_snapshot",
                }
            )
            parent, _ = self.normalizer.normalize(parent_id, content, metadata, data_version)
            chunks.append(parent)
            for index, child_text in enumerate(
                    self._split(content, self.settings.policy_child_chunk_size,
                                self.settings.policy_child_chunk_overlap)
            ):
                child_metadata = {**metadata, "chunk_type": "child", "parent_id": parent_id}
                child_id = f"{parent_id}-child-{index}"
                child, _ = self.normalizer.normalize(child_id, child_text, child_metadata, data_version)
                chunks.append(child)

        for page in pages:
            for line in (item.strip() for item in page.text.splitlines() if item.strip()):
                title = self._law_title(line)
                if title:
                    flush_article()
                    law_name = title
                    article_id = ""
                    article_lines = []
                    continue
                match = ARTICLE_PATTERN.match(line)
                if match:
                    flush_article()
                    article_id = self._article_number(match.group(1))
                    article_page = page.page_number
                    article_lines = [line]
                    continue
                if article_id:
                    article_lines.append(line)
        flush_article()
        return chunks

    def _commentary_chunks(
        self,
        pages: list[PdfPage],
        source: PdfSource,
        data_version: str,
    ) -> list[NormalizedChunk]:
        chunks = []
        for page in pages:
            metadata = self._metadata(source, page.page_number)
            metadata.update({"chunk_type": "document", "validity_status": "source_snapshot"})
            for index, content in enumerate(
                self._split(page.text, self.settings.pdf_chunk_size, self.settings.pdf_chunk_overlap)
            ):
                record_id = self._stable_id(source.path, page.page_number, index, content)
                chunk, _ = self.normalizer.normalize(record_id, content, metadata, data_version)
                chunks.append(chunk)
        return chunks

    @staticmethod
    def _metadata(source: PdfSource, page_number: int) -> dict:
        return {
            "title": source.title,
            "source": source.title,
            "source_path": source.path,
            "source_kind": source.source_kind,
            "source_as_of": source.source_as_of,
            "page_number": page_number,
            "authority_level": source.authority_level,
            "freshness_level": source.freshness_level,
            "region": "national",
        }

    @staticmethod
    def _law_title(line: str) -> str:
        candidate = re.sub(r"[（(].*$", "", line).strip()
        if 4 <= len(candidate) <= 50 and LAW_TITLE_PATTERN.fullmatch(candidate):
            return candidate
        return ""

    @staticmethod
    def _article_number(value: str) -> str:
        if value.isdigit():
            return value
        digits = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
        units = {"十": 10, "百": 100, "千": 1000}
        total = 0
        current = 0
        for char in value:
            if char in digits:
                current = digits[char]
            elif char in units:
                total += (current or 1) * units[char]
                current = 0
        return f"{total + current}"

    @staticmethod
    def _normalize_text(text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text)
        normalized = re.sub(r"[ \t]+", " ", normalized)
        return re.sub(r"\n{3,}", "\n\n", normalized).strip()

    @staticmethod
    def _split(text: str, chunk_size: int, overlap: int) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end].strip())
            if end == len(text):
                break
            start = end - overlap
        return chunks

    @staticmethod
    def _stable_id(*parts) -> str:
        value = "|".join(f"{part}" for part in parts)
        return hashlib.sha256(value.encode("utf-8")).hexdigest()