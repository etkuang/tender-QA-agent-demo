import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz

from agent_layer.domain.retrieval.chinese_number import chinese_number_converter
from agent_layer.infrastructure.cache.bm25_cache import BM25Cache
from agent_layer.infrastructure.config.agent_settings import settings
from agent_layer.infrastructure.vector.chroma_store import ChromaStore


CHAPTER_PATTERN = re.compile(r"第([一二三四五六七八九十百千\d]+)章\s*(.*)")
ARTICLE_HEADER_PATTERN = re.compile(r"第([一二三四五六七八九十百千\d]+)条[^\n]*")
ARTICLE_BOUNDARY_PATTERN = re.compile(r"第[一二三四五六七八九十百千\d]+条")
SUBSECTION_PATTERN = re.compile(r"[（(]([一二三四五六七八九十\d]+)[）)]")

DOC_TITLE_PATTERNS = [
    re.compile(r"^中华人民共和国.+[法条例]$"),
    re.compile(r"^关于.{4,}[通知意见函]$"),
    re.compile(r"^[^\s，。；！？、：（）\(\)\d\w]{4,14}(法|条例|办法|细则)$"),
]

TITLE_BLACKLIST = [
    "第", "条", "款", "项", "目", "节",
    "目录", "索引", "附录", "前言",
    "出版", "ISBN", "CIP",
    "违反", "不得", "应当", "可以", "必须",
    "规定", "处理", "处罚",
]


@dataclass
class ArticleInfo:
    article_id: int
    article_text: str
    title_hint: str = ""
    content: str = ""
    subsections: List[str] = field(default_factory=list)


@dataclass
class ChapterInfo:
    chapter_id: int
    chapter_text: str
    articles: List[ArticleInfo] = field(default_factory=list)


@dataclass
class LawDocument:
    law_name: str
    chapters: List[ChapterInfo] = field(default_factory=list)
    preamble: List[ArticleInfo] = field(default_factory=list)


class LegalStructureParser:
    TOC_KEYWORDS = ["目录", "编辑出版说明", "前言", "编写说明", "出版说明"]
    SUPPLEMENTARY_NAMES = ["附则", "附 则", "补则"]

    def __init__(self):
        self._cn = chinese_number_converter

    def parse(self, text: str, pdf_name: str = "") -> List[LawDocument]:
        if not text:
            return []

        documents = self._split_by_document_title(text)
        result = []

        for doc_title, doc_content in documents:
            if self._should_skip(doc_title):
                continue
            law_doc = self._parse_single_law(doc_title, doc_content)
            if law_doc and law_doc.chapters:
                result.append(law_doc)

        return result

    def _split_by_document_title(self, text: str) -> List[Tuple[str, str]]:
        lines = text.split("\n")
        documents = []
        current_doc = {"title": "", "lines": []}
        found_first_title = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            if self._is_document_title(stripped):
                if self._should_skip(stripped):
                    continue

                if found_first_title and current_doc["lines"]:
                    content = "\n".join(current_doc["lines"])
                    if len(content) > 200:
                        documents.append((current_doc["title"], content))
                elif not found_first_title:
                    found_first_title = True

                current_doc = {"title": stripped, "lines": []}
            elif found_first_title:
                current_doc["lines"].append(stripped)

        if found_first_title and current_doc["lines"]:
            content = "\n".join(current_doc["lines"])
            if len(content) > 200:
                documents.append((current_doc["title"], content))

        return documents

    def _is_document_title(self, line: str) -> bool:
        if len(line) < 8 or len(line) > 40:
            return False
        if re.match(r"^\d+$", line):
            return False
        if CHAPTER_PATTERN.match(line):
            return False
        if ARTICLE_BOUNDARY_PATTERN.match(line):
            return False

        for kw in TITLE_BLACKLIST:
            if kw in line:
                return False

        for kw in self.TOC_KEYWORDS:
            if kw in line:
                return False

        for pattern in DOC_TITLE_PATTERNS:
            if pattern.search(line):
                return True

        return False

    def _should_skip(self, title: str) -> bool:
        for kw in self.TOC_KEYWORDS:
            if kw in title:
                return True
        return False

    def _parse_single_law(self, law_name: str, content: str) -> LawDocument:
        law = LawDocument(law_name=law_name)
        content = self._remove_toc_block(content)
        chapter_blocks = self._split_by_chapter(content)

        for ch_text, ch_title, articles_text in chapter_blocks:
            chapter = self._parse_chapter(ch_text, ch_title, articles_text)
            if chapter:
                law.chapters.append(chapter)

        return law

    def _remove_toc_block(self, content: str) -> str:
        lines = content.split("\n")
        toc_start = -1
        first_chapter = -1

        for i, line in enumerate(lines):
            stripped = line.strip()
            if toc_start < 0 and ("目" in stripped and "录" in stripped and len(stripped) < 10):
                toc_start = i
            if CHAPTER_PATTERN.search(stripped):
                first_chapter = i
                break

        if toc_start >= 0 and first_chapter > toc_start:
            return "\n".join(lines[first_chapter:])

        return content

    def _split_by_chapter(self, content: str) -> List[Tuple[str, str, str]]:
        parts = re.split(r"(第[一二三四五六七八九十百千\d]+章[^\n]*)", content)
        result = []

        i = 0
        while i < len(parts) and not CHAPTER_PATTERN.search(parts[i]):
            i += 1

        while i < len(parts):
            header = parts[i].strip()
            body = parts[i + 1].strip() if i + 1 < len(parts) else ""
            i += 2

            match = CHAPTER_PATTERN.search(header)
            if match:
                ch_num = match.group(1)
                ch_name = header
                result.append((ch_num, ch_name, body))

        return result

    def _parse_chapter(self, ch_num_text: str, ch_title: str, articles_text: str) -> Optional[ChapterInfo]:
        ch_id = self._to_int(ch_num_text)
        chapter = ChapterInfo(chapter_id=ch_id, chapter_text=ch_title)

        articles = self._split_by_article(articles_text)
        for art_header, art_content in articles:
            art_info = self._parse_article(art_header, art_content)
            if art_info and len(art_info.content) >= 10:
                chapter.articles.append(art_info)

        if not chapter.articles:
            return None

        return chapter

    def _split_by_article(self, text: str) -> List[Tuple[str, str]]:
        if not text:
            return []

        first_match = ARTICLE_BOUNDARY_PATTERN.search(text)
        if not first_match:
            return []

        text = text[first_match.start():]
        parts = re.split(r"(第[一二三四五六七八九十百千\d]+条[^\n]*)", text)

        result = []
        i = 0
        while i < len(parts):
            if ARTICLE_HEADER_PATTERN.match(parts[i].strip()):
                header = parts[i].strip()
                body = parts[i + 1].strip() if i + 1 < len(parts) else ""
                i += 2
                result.append((header, body))
            else:
                i += 1

        return result

    def _parse_article(self, header: str, content: str) -> Optional[ArticleInfo]:
        match = ARTICLE_HEADER_PATTERN.search(header)
        if not match:
            return None

        art_num_text = match.group(1)
        art_id = self._to_int(art_num_text)
        art_text = match.group(0).strip()

        title_hint = ""
        hint_match = re.search(r"【([^】]+)】", header)
        if hint_match:
            title_hint = hint_match.group(1)

        full_content = f"{art_text}\n{content}" if content else art_text
        subsections = self._extract_subsections(content)

        return ArticleInfo(
            article_id=art_id,
            article_text=art_text,
            title_hint=title_hint,
            content=full_content,
            subsections=subsections,
        )

    def _extract_subsections(self, content: str) -> List[str]:
        if not content:
            return []

        parts = SUBSECTION_PATTERN.split(content)
        if len(parts) <= 1:
            return []

        subsections = []
        i = 1
        while i < len(parts):
            num = parts[i]
            body = parts[i + 1].strip() if i + 1 < len(parts) else ""
            i += 2
            subsections.append(f"({num}) {body}")

        return subsections

    def _to_int(self, num_text: str) -> int:
        text = num_text.strip()
        if text.isdigit():
            return int(text)

        arabic_str = self._cn.to_arabic(text)
        try:
            return int(arabic_str)
        except (ValueError, TypeError):
            return 0

    def get_statistics(self, documents: List[LawDocument]) -> Dict:
        total_chapters = 0
        total_articles = 0
        total_subsections = 0

        for doc in documents:
            total_chapters += len(doc.chapters)
            for chapter in doc.chapters:
                total_articles += len(chapter.articles)
                for article in chapter.articles:
                    total_subsections += len(article.subsections)

        return {
            "law_count": len(documents),
            "chapter_count": total_chapters,
            "article_count": total_articles,
            "subsection_count": total_subsections,
            "law_names": [doc.law_name for doc in documents],
        }


class ParentChunkBuilder:
    def __init__(self, source: str = "", header_injection: bool = True):
        self.source = source
        self.header_injection = header_injection
        self._counter = 0

    def build(self, documents: List[LawDocument]) -> List[Dict]:
        all_parents = []
        for doc in documents:
            all_parents.extend(self._build_for_law(doc))
        return all_parents

    def _build_for_law(self, law: LawDocument) -> List[Dict]:
        parents = []

        for article in law.preamble:
            chunk = self._build_parent(law.law_name, "前言/附则", article)
            if chunk:
                parents.append(chunk)

        for chapter in law.chapters:
            for article in chapter.articles:
                chunk = self._build_parent(law.law_name, chapter.chapter_text, article)
                if chunk:
                    parents.append(chunk)

        return parents

    def _build_parent(self, law_name: str, chapter: str, article: ArticleInfo) -> Dict:
        content = article.content
        chunk_id = self._make_chunk_id(law_name, article.article_id, content)
        full_header = self._make_header(law_name, chapter, article.article_text)

        if len(content) < 200:
            embedding_header = self._make_compact_header(law_name, chapter, article.article_text)
        else:
            embedding_header = full_header

        store_text = f"{embedding_header}\n{content}" if self.header_injection else content

        return {
            "chunk_id": chunk_id,
            "chunk_type": "parent",
            "parent_id": "",
            "law_name": law_name,
            "chapter": chapter,
            "article": article.article_text,
            "article_id": article.article_id,
            "content": store_text,
            "header_for_embedding": full_header,
            "raw_content": content,
            "metadata": {
                "source": self.source,
                "law_name": law_name,
                "chapter": chapter,
                "article": article.article_text,
                "article_id": article.article_id,
                "chunk_type": "parent",
                "parent_id": "",
                "type": "pdf_regulation",
            },
        }

    def _make_chunk_id(self, law_name: str, article_id: int, content: str = "") -> str:
        safe_name = law_name.replace(" ", "").replace("/", "_")[:20]
        self._counter += 1
        content_hash = abs(hash(content)) % 10000
        return f"parent_{safe_name}_{article_id}_{self._counter}_{content_hash}"

    def _make_header(self, law_name: str, chapter: str, article_text: str) -> str:
        parts = [f"《{law_name}》"]
        if chapter:
            parts.append(chapter)
        parts.append(article_text)
        return "\n".join(parts)

    def _make_compact_header(self, law_name: str, chapter: str, article_text: str) -> str:
        short_name = law_name.replace("中华人民共和国", "").replace("法律法规全书", "法规全书")
        parts = [f"《{short_name}》"]
        if chapter:
            parts.append(chapter)
        parts.append(article_text)
        return " / ".join(parts)


class ChildChunkBuilder:
    def __init__(self):
        self.threshold = getattr(settings, "legal_child_split_threshold", 400)
        self.target_size = getattr(settings, "legal_child_target_size", 300)
        self.overlap = getattr(settings, "legal_child_overlap", 40)
        self.header_enabled = getattr(settings, "legal_header_injection_enabled", True)

    def build_all(self, parents: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        searchable_parents = []
        all_children = []

        for parent in parents:
            raw = parent.get("raw_content", "")
            if len(raw) <= self.threshold:
                searchable_parents.append(parent)
            else:
                children = self._build_for_parent(parent)
                if children:
                    all_children.extend(children)
                else:
                    searchable_parents.append(parent)

        return searchable_parents, all_children

    def _build_for_parent(self, parent: Dict) -> List[Dict]:
        raw = parent.get("raw_content", "")
        if len(raw) <= self.threshold:
            return []

        for strategy in [self._split_by_subsections, self._split_by_paragraphs, self._split_by_sentences]:
            segments = strategy(raw)
            if len(segments) > 1:
                return self._make_children(segments, parent)

        return []

    def _split_by_subsections(self, content: str) -> List[str]:
        pattern = re.compile(r"[（(]([一二三四五六七八九十\d]+)[）)]")
        matches = list(pattern.finditer(content))
        if len(matches) < 2:
            return [content]

        segments = []
        first_cut = matches[0].start()
        if first_cut > 10:
            segments.append(content[:first_cut].strip())

        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            segment = content[start:end].strip()
            if len(segment) > 20:
                segments.append(segment)

        if len(segments) <= 1:
            return [content]

        return self._merge_long_segments(segments, self.target_size)

    def _split_by_paragraphs(self, content: str) -> List[str]:
        paragraphs = re.split(r"\n\s*\n", content)
        paragraphs = [item.strip() for item in paragraphs if item.strip()]

        if len(paragraphs) <= 1:
            return [content]

        merged = []
        current = ""
        for paragraph in paragraphs:
            if len(current) + len(paragraph) < self.target_size:
                current = current + "\n\n" + paragraph if current else paragraph
            else:
                if current:
                    merged.append(current)
                current = paragraph
        if current:
            merged.append(current)

        return merged if len(merged) > 1 else [content]

    def _split_by_sentences(self, content: str) -> List[str]:
        sentences = re.split(r"(?<=[。；！？])", content)
        sentences = [item.strip() for item in sentences if item.strip()]

        if len(sentences) <= 1:
            sentences = re.split(r"(?<=[，,])", content)
            sentences = [item.strip() for item in sentences if item.strip()]

        if len(sentences) <= 1:
            return [content]

        chunks = []
        current = ""
        for sentence in sentences:
            if len(current) + len(sentence) < self.target_size:
                current = current + sentence if current else sentence
            else:
                if current:
                    chunks.append(current)
                if chunks and self.overlap > 0:
                    previous = chunks[-1]
                    overlap_text = previous[-self.overlap:] if len(previous) > self.overlap else previous
                    current = overlap_text + sentence
                else:
                    current = sentence
        if current:
            chunks.append(current)

        return chunks if len(chunks) > 1 else [content]

    def _merge_long_segments(self, segments: List[str], max_size: int) -> List[str]:
        result = []
        current = ""

        for segment in segments:
            if len(segment) > max_size * 2:
                if current:
                    result.append(current)
                    current = ""
                result.extend(self._split_by_paragraphs(segment))
            elif len(current) + len(segment) < max_size:
                current = current + "\n" + segment if current else segment
            else:
                if current:
                    result.append(current)
                current = segment

        if current:
            result.append(current)

        return result if result else segments

    def _make_children(self, segments: List[str], parent: Dict) -> List[Dict]:
        children = []
        parent_id = parent["chunk_id"]
        header = parent.get("header_for_embedding", "")

        for i, segment in enumerate(segments):
            if len(segment) < 20:
                continue

            chunk_id = f"{parent_id}_child{i}"
            store_text = f"{header}\n{segment}" if self.header_enabled and header else segment

            children.append({
                "chunk_id": chunk_id,
                "chunk_type": "child",
                "parent_id": parent_id,
                "law_name": parent["law_name"],
                "chapter": parent["chapter"],
                "article": parent["article"],
                "article_id": parent["article_id"],
                "content": store_text,
                "raw_content": segment,
                "header_for_embedding": header,
                "metadata": {
                    "source": parent["metadata"].get("source", ""),
                    "law_name": parent["law_name"],
                    "chapter": parent["chapter"],
                    "article": parent["article"],
                    "article_id": parent["article_id"],
                    "chunk_type": "child",
                    "parent_id": parent_id,
                    "type": "pdf_regulation",
                },
            })

        return children


def sliding_window_chunk(text: str, chunk_size: int = 500, overlap: int = 100, source_label: str = "") -> List[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            for sep in ["。", "\n", "；", "！", "？"]:
                last_sep = text.rfind(sep, start, end)
                if last_sep > start + chunk_size // 2:
                    end = last_sep + 1
                    break

        chunk = text[start:end].strip()
        if chunk:
            if source_label:
                chunk = f"【{source_label}】\n{chunk}"
            chunks.append(chunk)

        start = end - overlap if end < len(text) else end

    return chunks


def extract_full_text_from_pdf(pdf_path: str) -> Tuple[str, int]:
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

    doc = fitz.open(pdf_path)
    full_text_parts = []
    total_pages = len(doc)

    for page in doc:
        page_text = page.get_text()
        if page_text:
            full_text_parts.append(page_text)

    doc.close()

    full_text = "\n".join(full_text_parts)
    print(f"  提取了 {total_pages} 页，共 {len(full_text)} 字符")
    return full_text, total_pages


def load_manual_eval_set() -> List[Dict]:
    eval_path = Path(settings.manual_eval_path)
    if not eval_path.exists():
        return []

    with open(eval_path, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_pdfs() -> List[Dict]:
    pdf_dir = settings.pdf_dir_path
    pdf_files = []

    print(f"\n📂 PDF目录: {settings.pdf_dir_path}")
    print(f"💾 存储目录: {settings.chroma_persist_dir_path}")

    if not pdf_dir.exists():
        print(f"⚠️ PDF目录不存在: {settings.pdf_dir_path}")
        return []

    metadata_map = settings.pdf_metadata

    for pdf_file in pdf_dir.glob("*.pdf"):
        stem = pdf_file.stem

        if stem in metadata_map:
            meta = metadata_map[stem]
            author = meta.get("author", settings.default_author)
            chunk_mode = meta.get("chunk_mode", settings.default_chunk_mode)
        else:
            chunk_mode = settings.default_chunk_mode
            author = settings.default_author
            print(f"  ⚠️ 未在 pdf_metadata 中配置 '{stem}'，回退为默认模式: {chunk_mode}")

        pdf_files.append({
            "path": str(pdf_file),
            "name": stem,
            "author": author,
            "chunk_size": settings.default_chunk_size,
            "overlap": settings.default_overlap,
            "chunk_mode": chunk_mode,
        })

    print(f"📁 发现 {len(pdf_files)} 个PDF文件")
    for pdf in pdf_files:
        if pdf["chunk_mode"] == "law_article":
            mode_desc = "Parent-Child结构化法条切块"
        else:
            mode_desc = "滑动窗口切块"
        print(f"    - {pdf['name']} → {mode_desc}")

    return pdf_files


def chunk_by_structure(full_text: str, pdf_name: str) -> Optional[Dict]:
    parser = LegalStructureParser()
    documents = parser.parse(full_text, pdf_name)
    doc_stats = parser.get_statistics(documents)

    print(
        f"  [统计] 解析结果: {doc_stats['law_count']} 部法律, "
        f"{doc_stats['chapter_count']} 章, {doc_stats['article_count']} 条"
    )

    if doc_stats["article_count"] == 0:
        print("  [警告] 未解析到法条，回退到滑动窗口切块")
        return None

    parent_builder = ParentChunkBuilder(source=pdf_name, header_injection=True)
    parents = parent_builder.build(documents)

    child_builder = ChildChunkBuilder()
    searchable_parents, children = child_builder.build_all(parents)

    all_chunks = []
    all_metadatas = []
    all_ids = []

    for parent in searchable_parents:
        all_chunks.append(parent["content"])
        all_metadatas.append(parent["metadata"])
        all_ids.append(parent["chunk_id"])

    for child in children:
        all_chunks.append(child["content"])
        all_metadatas.append(child["metadata"])
        all_ids.append(child["chunk_id"])

    searchable_ids = {item["chunk_id"] for item in searchable_parents}
    parent_only_chunks = []
    parent_only_metadatas = []
    parent_only_ids = []

    for parent in parents:
        if parent["chunk_id"] not in searchable_ids:
            parent_only_chunks.append(parent["content"])
            parent_only_metadatas.append(parent["metadata"])
            parent_only_ids.append(parent["chunk_id"])

    stats = {
        "law_count": doc_stats["law_count"],
        "chapter_count": doc_stats["chapter_count"],
        "article_count": doc_stats["article_count"],
        "parent_count": len(parents),
        "child_count": len(children),
        "searchable_parent_count": len(searchable_parents),
        "total_searchable": len(all_chunks),
        "parent_only_count": len(parent_only_chunks),
    }

    return {
        "searchable_chunks": all_chunks,
        "searchable_metadatas": all_metadatas,
        "searchable_ids": all_ids,
        "parent_only_chunks": parent_only_chunks,
        "parent_only_metadatas": parent_only_metadatas,
        "parent_only_ids": parent_only_ids,
        "stats": stats,
    }


def process_pdf_to_chroma(pdf_config: Dict, client: ChromaStore):
    pdf_path = pdf_config["path"]
    pdf_name = pdf_config["name"]
    author = pdf_config.get("author", settings.default_author)
    chunk_mode = pdf_config.get("chunk_mode", settings.default_chunk_mode)
    chunk_size = pdf_config.get("chunk_size", settings.default_chunk_size)
    overlap = pdf_config.get("overlap", settings.default_overlap)

    print(f"\n处理: {pdf_name} (切块模式: {chunk_mode})")

    if not Path(pdf_path).exists():
        print(f"  跳过：文件不存在 -> {pdf_path}")
        return

    full_text, total_pages = extract_full_text_from_pdf(pdf_path)
    if not full_text or len(full_text) < 100:
        print("  跳过：文本内容不足")
        return

    if chunk_mode == "law_article":
        result = chunk_by_structure(full_text, pdf_name)

        if result is None:
            chunks = sliding_window_chunk(full_text, chunk_size, overlap, source_label=pdf_name)
            metadatas = []
            ids = []

            for i, chunk in enumerate(chunks):
                metadatas.append({
                    "source": pdf_name,
                    "author": author,
                    "chunk_index": i,
                    "total_pages": total_pages,
                    "type": "pdf_regulation",
                    "chunk_type": "sliding",
                    "parent_id": "",
                })
                ids.append(f"reg_{pdf_name}_{i}_{abs(hash(chunk)) % 10000}")

            if chunks:
                client.add_documents("regulations", chunks, metadatas, ids)
                print(f"  [完成] 回退滑窗后入库 {len(chunks)} 条")
            return

        stats = result["stats"]
        print(
            f"  [统计] 可检索 {stats['total_searchable']} 条 "
            f"({stats['searchable_parent_count']} parent + {stats['child_count']} child)"
        )

        if result["searchable_chunks"]:
            client.add_documents(
                "regulations",
                result["searchable_chunks"],
                result["searchable_metadatas"],
                result["searchable_ids"],
            )
            print(f"  [完成] 已入库可检索 chunk: {len(result['searchable_chunks'])} 条")

        if result["parent_only_chunks"]:
            client.add_documents(
                "regulations",
                result["parent_only_chunks"],
                result["parent_only_metadatas"],
                result["parent_only_ids"],
            )
            print(f"  [完成] 已入库 parent-only chunk: {len(result['parent_only_chunks'])} 条")

        return

    chunks = sliding_window_chunk(full_text, chunk_size, overlap, source_label=pdf_name)
    metadatas = []
    ids = []

    for i, chunk in enumerate(chunks):
        metadatas.append({
            "source": pdf_name,
            "author": author,
            "chunk_index": i,
            "total_pages": total_pages,
            "type": "pdf_regulation",
            "chunk_type": "sliding",
            "parent_id": "",
        })
        ids.append(f"reg_{pdf_name}_{i}_{abs(hash(chunk)) % 10000}")

    if not chunks:
        print("  警告：无有效chunk")
        return

    client.add_documents("regulations", chunks, metadatas, ids)
    print(f"  [完成] 已入库 {len(chunks)} 条")


def main():
    print("=" * 60)
    print("PDF知识库初始化工具 (Reference Parent-Child方式)")
    print("=" * 60)
    print("策略: law_article -> Parent-Child结构化切块")
    print("      sliding     -> 滑动窗口切块")
    print("=" * 60)

    print(f"\n📂 PDF目录: {settings.pdf_dir}")
    print(f"💾 存储目录: {settings.chroma_persist_dir}")
    print("=" * 60)

    client = ChromaStore()

    print("\n🗑️ 清空现有 regulations 库...")
    client.delete_collection("regulations")
    BM25Cache().invalidate("regulations")
    print("🗑️ BM25 缓存已失效")

    pdf_files = discover_pdfs()

    if not pdf_files:
        print("❌ 未找到PDF文件，请检查目录")
        print(f"   期望目录: {settings.pdf_dir_path}")
        return

    for pdf_config in pdf_files:
        process_pdf_to_chroma(pdf_config, client)

    print("\n" + "=" * 60)
    print("✅ 初始化完成")
    print("=" * 60)
    print(f"\nRegulations库总计数: {client.get_count('regulations')} 条")

    eval_set = load_manual_eval_set()
    if eval_set:
        print(f"\n📋 手工评估集: {len(eval_set)} 条")
        print(f"   位置: {settings.manual_eval_file}")
    else:
        print("\n💡 提示：如需评估准确率，请创建手工评估集文件")
        print(f"   {settings.manual_eval_file}")
        print("   格式: [{\"question\": \"...\", \"expected_answer\": \"...\"}]")
    print("=" * 60)


if __name__ == "__main__":
    main()