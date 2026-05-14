#!/usr/bin/env python
"""准确率测试脚本 - 验证Top-5准确率是否达到92%"""

import asyncio
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.core.retriever import HybridRetriever
from app.core.router import IntentRouter

MANUAL_EVAL_PATH = "./data/manual_eval_set.json"


async def evaluate():
    print("=" * 60)
    print("准确率测试")
    print("=" * 60)

    # 加载评估集
    if not Path(MANUAL_EVAL_PATH).exists():
        print(f"\n错误: 评估集文件不存在: {MANUAL_EVAL_PATH}")
        print("请创建手工评估集后重试")
        return

    with open(MANUAL_EVAL_PATH, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)

    print(f"\n加载评估集: {len(test_cases)} 条")

    retriever = HybridRetriever()
    router = IntentRouter()

    correct = 0
    total = len(test_cases)
    failed = []

    for i, case in enumerate(test_cases, 1):
        question = case.get("question", "")
        expected_answer = case.get("expected_answer", "")

        print(f"\n[{i}/{total}] 测试: {question[:50]}...")

        # 意图识别
        route = router.route(question)

        # 检索
        results = retriever.search(
            query=question,
            collection=route["collection"],
            top_k=5
        )

        # 验证（简化的关键词匹配，实际可用更复杂的评估）
        found = False
        for j, r in enumerate(results):
            text = r.get("text", "")
            # 检查答案中的关键词是否在检索结果中
            keywords = expected_answer.split()[:3] if expected_answer else []
            if keywords and all(kw in text for kw in keywords):
                found = True
                print(f"  ✅ 第{j + 1}位命中")
                break
            elif not keywords and j == 0:
                found = True
                break

        if found:
            correct += 1
        else:
            print(f"  ❌ 未命中")
            failed.append(question)

    accuracy = correct / total * 100

    print("\n" + "=" * 60)
    print(f"测试结果:")
    print(f"  总用例: {total}")
    print(f"  正确: {correct}")
    print(f"  准确率: {accuracy:.1f}%")
    print(f"  目标: 92%")
    print(f"  {'✅ 达标' if accuracy >= 92 else '❌ 未达标'}")
    print("=" * 60)

    if failed:
        print(f"\n失败用例 ({len(failed)}):")
        for q in failed[:10]:
            print(f"  - {q[:60]}...")


if __name__ == "__main__":
    asyncio.run(evaluate())
