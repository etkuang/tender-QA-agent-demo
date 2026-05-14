#!/usr/bin/env python
"""PDF知识库初始化 - 处理三本招标法规类PDF"""

import sys
import json
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent))

import fitz
from app.storage.chroma_store import ChromaStore
from app.config import settings

PDF_FILES = [
    {
        "path": "./data/pdfs/招标投标法律解读与风险防范实务 (白如银) (Z-Library).pdf",
        "name": "招标投标法律解读与风险防范实务",
        "author": "白如银",
        "chunk_size": 500,
        "overlap": 100
    },
    {
        "path": "./data/pdfs/政府采购、工程招标、投标与评标1200问（第3版）_刘海桑 编著.pdf",
        "name": "政府采购、工程招标、投标与评标1200问",
        "author": "刘海桑",
        "chunk_size": 500,
        "overlap": 100
    },
    {
        "path": "./data/pdfs/中华人民共和国招标投标法律法规全书：含相关政策 (法律法规出版社法规中心) (Z-Library)(OCR).pdf",
        "name": "中华人民共和国招标投标法律法规全书",
        "author": "中国法制出版社",
        "chunk_size": 500,
        "overlap": 100
    }
]

MANUAL_EVAL_PATH = "./data/manual_eval_set.json"


def sliding_window_chunk(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """滑动窗口切块"""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # 尝试在句号、换行处断开
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


def extract_full_text_from_pdf(pdf_path: str) -> tuple:
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
    if not Path(MANUAL_EVAL_PATH).exists():
        return []

    with open(MANUAL_EVAL_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def process_pdf_to_chroma(pdf_config: Dict, client: ChromaStore):
    """处理PDF并入Chroma"""
    pdf_path = pdf_config["path"]
    pdf_name = pdf_config["name"]
    author = pdf_config.get("author", "")
    chunk_size = pdf_config.get("chunk_size", 500)
    overlap = pdf_config.get("overlap", 100)

    print(f"\n处理: {pdf_name}")

    if not Path(pdf_path).exists():
        print(f"  跳过：文件不存在 -> {pdf_path}")
        return

    full_text, total_pages = extract_full_text_from_pdf(pdf_path)
    if not full_text or len(full_text) < 100:
        print(f"  跳过：文本内容不足")
        return

    chunks = sliding_window_chunk(full_text, chunk_size, overlap)
    print(f"  切块: {len(chunks)} 条")

    if not chunks:
        print(f"  警告：无有效chunk")
        return

    metadatas = []
    ids = []
    for i, chunk in enumerate(chunks):
        metadatas.append({
            "source": pdf_name,
            "author": author,
            "chunk_index": i,
            "total_pages": total_pages,
            "type": "pdf_regulation"
        })
        ids.append(f"reg_{pdf_name}_{i}_{hash(chunk) % 10000}")

    client.add_documents("regulations", chunks, metadatas, ids)
    print(f"  已入库 {len(chunks)} 条到 regulations 库")


def main():
    print("=" * 60)
    print("PDF知识库初始化工具")
    print("=" * 60)
    print("处理三本招标法规类PDF")
    print("策略：全文提取 → 滑动窗口切块（不按页切割）")
    print("=" * 60)

    client = ChromaStore()

    print("\n清空现有 regulations 库...")
    client.delete_collection("regulations")

    for pdf_config in PDF_FILES:
        process_pdf_to_chroma(pdf_config, client)

    print("\n" + "=" * 60)
    print("初始化完成!")
    print(f"  Regulations库: {client.get_count('regulations')} 条")
    print("=" * 60)

    eval_set = load_manual_eval_set()
    if eval_set:
        print(f"\n手工评估集: {len(eval_set)} 条")
        print(f"  位置: {MANUAL_EVAL_PATH}")
    else:
        print(f"\n提示：如需评估准确率，请创建手工评估集文件:")
        print(f"  {MANUAL_EVAL_PATH}")
        print(f"  格式: [{{\"question\": \"...\", \"expected_answer\": \"...\"}}]")


if __name__ == "__main__":
    main()
