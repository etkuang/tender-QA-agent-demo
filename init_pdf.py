#!/usr/bin/env python
"""PDF知识库初始化 - 处理招标法规类PDF（配置化版本）"""

import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import fitz
from app.storage.chroma_store import ChromaStore
from config import settings
from app.utils.chinese_number import chinese_number_converter  # ← 修改导入
from app.core.bm25_cache import BM25Cache

# ========== 删除原有的 chinese_to_arabic_dynamic 函数 ==========
# 改为使用 app.utils.chinese_number 中的版本


def extract_article_number(header: str) -> str:
    """从法条标题中提取编号（使用配置化的转换器）"""
    # 先匹配阿拉伯数字
    match = re.search(r'第\s*(\d+)\s*条', header)
    if match:
        return match.group(1)

    # 匹配中文数字（使用配置化的转换器）
    match = re.search(r'第([一二三四五六七八九十百千万零]+)条', header)
    if match:
        return chinese_number_converter.to_arabic(match.group(1))

    return "unknown"


def is_document_title(line: str) -> bool:
    """
    判断一行是否为文件标题
    规则从配置读取
    """
    line = line.strip()
    if not line or len(line) < 5:
        return False

    # 排除页码、空行等
    if re.match(r'^\d+$', line):
        return False

    # 以"第X条"开头的不是标题，是法条
    if re.match(r'^第[一二三四五六七八九十百千万\d]+条', line):
        return False

    # 从配置读取标题匹配模式
    title_patterns = settings.document_title_patterns
    for pattern in title_patterns:
        if re.search(pattern, line):
            return True

    return False


def should_exclude_document(doc_title: str) -> Tuple[bool, str]:
    """
    判断是否应该排除某个文档
    返回 (是否排除, 排除原因)
    规则从配置读取
    """
    # 检查排除关键词
    for kw in settings.exclude_doc_keywords:
        if kw in doc_title:
            return True, f"匹配排除关键词: {kw}"

    # 检查排除内容关键词
    for kw in settings.exclude_content_keywords:
        if kw in doc_title:
            return True, f"匹配排除内容: {kw}"

    return False, ""


def get_document_chunk_mode(doc_title: str) -> str:
    """
    判断文档应该使用哪种切块模式
    返回 "split_by_article" 或 "keep_as_whole"
    规则从配置读取
    """
    # 检查是否需要按法条切分
    for kw in settings.split_by_article_keywords:
        if kw in doc_title:
            return "split_by_article"

    # 检查是否需要整体保留
    for kw in settings.keep_as_whole_keywords:
        if kw in doc_title:
            return "keep_as_whole"

    # 默认：整体保留（通知、意见等）
    return "keep_as_whole"


def sliding_window_chunk(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """滑动窗口切块"""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            for sep in ['。', '\n', '；', '！', '？']:
                last_sep = text.rfind(sep, start, end)
                if last_sep > start + chunk_size // 2:
                    end = last_sep + 1
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap if end < len(text) else end

    return chunks


def chunk_by_document(text: str, pdf_name: str) -> List[Tuple[str, str, str]]:
    """
    按文件标题切分，返回 [(chunk_text, doc_title, article_num), ...]

    对于有法条的文件（如招标投标法），按"第X条"切分
    对于无法条的文件（如通知、意见），整个文件作为一个chunk
    """
    if not text:
        return []

    # ========== 第一步：按文件标题切分 ==========
    lines = text.split('\n')
    documents = []
    current_doc = {"title": "", "content": []}

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if is_document_title(line):
            # 保存上一个文档
            if current_doc["content"]:
                documents.append((current_doc["title"], '\n'.join(current_doc["content"])))
            # 开始新文档
            current_doc = {"title": line, "content": []}
        else:
            current_doc["content"].append(line)

    # 保存最后一个文档
    if current_doc["content"]:
        documents.append((current_doc["title"], '\n'.join(current_doc["content"])))

    print(f"  📄 按文件切分: 共 {len(documents)} 个文件")

    # ========== 第二步：过滤和切分 ==========
    result = []
    filtered_count = 0
    kept_as_whole_count = 0
    split_by_article_count = 0

    for doc_title, doc_content in documents:
        # 检查是否排除
        exclude, reason = should_exclude_document(doc_title)
        if exclude:
            filtered_count += 1
            print(f"    [排除] {doc_title} -> {reason}")
            continue

        # 确定切块模式
        chunk_mode = get_document_chunk_mode(doc_title)

        if chunk_mode == "split_by_article":
            # 按法条切分
            article_pattern = r'(第[一二三四五六七八九十百千万\d]+条[^\n]*)(.*?)(?=第[一二三四五六七八九十百千万\d]+条|$)'
            matches = re.findall(article_pattern, doc_content, re.DOTALL)

            if matches:
                for match in matches:
                    article_header = match[0].strip()
                    article_content = match[1].strip()
                    full_article = f"{article_header}\n{article_content}" if article_content else article_header

                    if len(full_article) < 30:
                        continue

                    article_num = extract_article_number(article_header)
                    result.append((full_article, doc_title, article_num))
                    split_by_article_count += 1

                if split_by_article_count <= 20:
                    print(f"    [法条] {doc_title} → 切出 {len(matches)} 条")
            else:
                # 无法条但按类型应该有的，作为整体保留
                if len(doc_content) > 100:
                    result.append((doc_content, doc_title, "full"))
                    kept_as_whole_count += 1
                    print(f"    [整体-降级] {doc_title}")
        else:
            # 不需要按法条切分的，整体保留
            if len(doc_content) > 100:
                result.append((doc_content, doc_title, "full"))
                kept_as_whole_count += 1
                if kept_as_whole_count <= 20:
                    print(f"    [整体] {doc_title}")

    print(
        f"  📊 最终结果: 法条类 {split_by_article_count} 条, 整体类 {kept_as_whole_count} 条, 过滤 {filtered_count} 个文件")

    return result


def chunk_by_law_articles(text: str, pdf_name: str, chunk_size: int = 500, overlap: int = 100) -> List[Tuple[str, str]]:
    """
    法律条文专用切块器，返回 [(chunk_text, article_num), ...]
    内部调用 chunk_by_document，然后扁平化结果
    """
    chunks_with_meta = chunk_by_document(text, pdf_name)
    result = []
    for chunk_text, doc_title, article_num in chunks_with_meta:
        result.append((chunk_text, article_num))
    return result


def chunk_by_page_fallback(text: str) -> List[str]:
    """备用方案：按自然段落切块"""
    chunks = []
    paragraphs = text.split('\n\n')
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) < 800:
            current_chunk += "\n" + para if current_chunk else para
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = para

    if current_chunk:
        chunks.append(current_chunk)

    print(f"  备用切块: 共 {len(chunks)} 条")
    return chunks


def extract_full_text_from_pdf(pdf_path: str) -> Tuple[str, int]:
    """提取PDF全文，返回 (full_text, total_pages)"""
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

    doc = fitz.open(pdf_path)
    full_text_parts = []
    total_pages = len(doc)

    for page_num, page in enumerate(doc):
        page_text = page.get_text()
        if page_text:
            full_text_parts.append(page_text)

    doc.close()

    full_text = "\n".join(full_text_parts)
    print(f"  提取了 {total_pages} 页，共 {len(full_text)} 字符")

    return full_text, total_pages


def load_manual_eval_set() -> List[Dict]:
    """加载手工整理的评估集"""
    eval_path = Path(settings.manual_eval_path)
    if not eval_path.exists():
        return []

    with open(eval_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def discover_pdfs() -> List[Dict]:
    """
    发现目录下的PDF文件，从 config.yaml 的 pdf_metadata 显式配置读取切分模式。
    每个PDF必须用精确文件名（不含.pdf）在 pdf_metadata 中配置。
    未配置的PDF会回退到默认模式并打印警告。
    """
    pdf_dir = Path(settings.pdf_dir)
    pdf_files = []

    if not pdf_dir.exists():
        print(f"⚠️ PDF目录不存在: {settings.pdf_dir}")
        return []

    metadata_map = settings.pdf_metadata  # {exact_stem: {author, chunk_mode}}

    for pdf_file in pdf_dir.glob("*.pdf"):
        stem = pdf_file.stem

        # 精确匹配 config.yaml 中的配置
        if stem in metadata_map:
            meta = metadata_map[stem]
            author = meta.get("author", settings.default_author)
            chunk_mode = meta.get("chunk_mode", settings.default_chunk_mode)
        else:
            chunk_mode = settings.default_chunk_mode
            author = settings.default_author
            print(f"  ⚠️ 未在 pdf_metadata 中配置 '{stem}'，回退为默认模式: {chunk_mode}")
            print(f"     请在 config.yaml → pdf → pdf_metadata 中添加该PDF的配置")

        pdf_files.append({
            "path": str(pdf_file),
            "name": stem,
            "author": author,
            "chunk_size": settings.default_chunk_size,
            "overlap": settings.default_overlap,
            "chunk_mode": chunk_mode
        })

    print(f"📁 发现 {len(pdf_files)} 个PDF文件")
    for pdf in pdf_files:
        mode_desc = "法条切分" if pdf["chunk_mode"] == "law_article" else "滑动窗口"
        print(f"    - {pdf['name']} → {mode_desc}")

    return pdf_files


def process_pdf_to_chroma(pdf_config: Dict, client: ChromaStore):
    """处理PDF并入Chroma"""
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
        print(f"  跳过：文本内容不足")
        return

    chunks = []
    metadatas = []

    if chunk_mode == "law_article":
        # 使用按文档切分方法
        chunks_with_meta = chunk_by_document(full_text, pdf_name)

        chunks = [c[0] for c in chunks_with_meta]

        for i, (chunk, doc_title, article_num) in enumerate(chunks_with_meta):
            metadatas.append({
                "source": pdf_name,
                "author": author,
                "chunk_index": i,
                "total_pages": total_pages,
                "type": "law_article",
                "chunk_mode": chunk_mode,
                "article_num": article_num,
                "doc_title": doc_title
            })
    else:  # sliding (default)
        chunks = sliding_window_chunk(full_text, chunk_size, overlap)
        for i, chunk in enumerate(chunks):
            metadatas.append({
                "source": pdf_name,
                "author": author,
                "chunk_index": i,
                "total_pages": total_pages,
                "type": "commentary",
                "chunk_mode": chunk_mode
            })

    print(f"  实际入库: {len(chunks)} 条")

    if not chunks:
        print(f"  警告：无有效chunk")
        return

    ids = [f"reg_{pdf_name}_{i}_{hash(chunks[i]) % 10000}" for i in range(len(chunks))]
    client.add_documents("regulations", chunks, metadatas, ids)
    print(f"  ✅ 已入库 {len(chunks)} 条到 regulations 库")


def main():
    print("=" * 60)
    print("PDF知识库初始化工具 (配置化版本)")
    print("=" * 60)

    print(f"\n📂 PDF目录: {settings.pdf_dir}")
    print(f"💾 存储目录: {settings.chroma_persist_dir}")
    print("=" * 60)

    client = ChromaStore()

    print("\n🗑️ 清空现有 regulations 库...")
    client.delete_collection("regulations")
    # 使 BM25 缓存失效，确保下次检索时重建索引
    BM25Cache().invalidate("regulations")
    print("🗑️ BM25 缓存已失效")

    stats = {
        "law_article": 0,
        "commentary": 0,
        "total": 0
    }

    # 自动发现PDF文件
    pdf_files = discover_pdfs()

    if not pdf_files:
        print("❌ 未找到PDF文件，请检查目录")
        print(f"   期望目录: {settings.pdf_dir}")
        return

    for pdf_config in pdf_files:
        before_count = client.get_count("regulations")
        process_pdf_to_chroma(pdf_config, client)
        after_count = client.get_count("regulations")
        added = after_count - before_count

        chunk_mode = pdf_config.get("chunk_mode", settings.default_chunk_mode)
        if chunk_mode == "law_article":
            stats["law_article"] += added
        else:
            stats["commentary"] += added
        stats["total"] += added

    print("\n" + "=" * 60)
    print("✅ 初始化完成!")
    print("=" * 60)
    print(f"\n📊 入库统计:")
    if stats["law_article"] > 0:
        print(f"   法条类 (law_article): {stats['law_article']} 条")
    if stats["commentary"] > 0:
        print(f"   论述类 (commentary):  {stats['commentary']} 条")
    print(f"   ────────────────────")
    print(f"   总计:                  {stats['total']} 条")
    print(f"\n  Regulations库总计数: {client.get_count('regulations')} 条")

    eval_set = load_manual_eval_set()
    if eval_set:
        print(f"\n📋 手工评估集: {len(eval_set)} 条")
        print(f"   位置: {settings.manual_eval_path}")
    else:
        print(f"\n💡 提示：如需评估准确率，请创建手工评估集文件:")
        print(f"   {settings.manual_eval_path}")
        print(f"   格式: [{{\"question\": \"...\", \"expected_answer\": \"...\"}}]")
    print("=" * 60)


if __name__ == "__main__":
    main()