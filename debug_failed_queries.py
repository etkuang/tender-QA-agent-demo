#!/usr/bin/env python
"""诊断哪些查询失败以及为什么"""
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from app.core.retriever import HybridRetriever
from config import settings


async def debug_failed_queries():
    print("=" * 70)
    print("🔍 诊断 Weighted 策略失败的查询")
    print("=" * 70)

    # 加载评估集
    eval_path = Path("./data/eval_set_25.json")
    with open(eval_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    if isinstance(raw_data, dict) and "questions" in raw_data:
        questions = raw_data["questions"]
    else:
        questions = raw_data

    # 保存原策略
    old_strategy = settings.fusion_strategy
    settings.fusion_strategy = "weighted"

    retriever = HybridRetriever()

    failed = []

    for i, item in enumerate(questions):
        print(f"\n[{i + 1}/25] {item['id']}: {item['question'][:50]}...")

        try:
            # 使用同步调用，方便调试
            results = retriever.search(
                query=item["question"],
                collection="regulations",
                top_k=10
            )

            if not results:
                print(f"   ❌ 失败: 返回空结果")
                failed.append({
                    "id": item["id"],
                    "question": item["question"],
                    "error": "空结果"
                })
            else:
                print(f"   ✅ 成功: 返回 {len(results)} 条")

        except Exception as e:
            print(f"   ❌ 异常: {e}")
            failed.append({
                "id": item["id"],
                "question": item["question"],
                "error": str(e)
            })

    # 恢复原策略
    settings.fusion_strategy = old_strategy

    print("\n" + "=" * 70)
    print(f"📊 诊断结果: {len(failed)} 条查询失败")
    print("=" * 70)

    for f in failed:
        print(f"\n❌ {f['id']}: {f['question']}")
        print(f"   错误: {f['error']}")

    return failed


if __name__ == "__main__":
    asyncio.run(debug_failed_queries())