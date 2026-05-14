#!/usr/bin/env python
"""准确率测试脚本 - 改进版，支持中文智能匹配"""

import asyncio
import sys
import json
import re
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from app.core.retriever import HybridRetriever
from app.core.router import IntentRouter

# 尝试导入 jieba，如果没有则使用简单的匹配
try:
    import jieba

    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    print("提示: jieba 未安装，使用简化匹配模式")


def extract_keywords(text: str) -> List[str]:
    """
    从期望答案中提取关键词

    例如:
    - "重庆稳信科技有限公司" -> ["重庆", "稳信", "科技"]
    - "串通投标 投标报价" -> ["串通投标", "投标报价"]
    - "407000" -> ["407000"]
    """
    if not text:
        return []

    # 如果包含数字金额，优先保留完整数字
    numbers = re.findall(r'\d+(?:\.\d+)?', text)
    if numbers:
        return numbers

    # 如果包含空格，按空格分割（用户手动指定的关键词）
    if ' ' in text:
        return [kw.strip() for kw in text.split() if len(kw.strip()) >= 2]

    # 使用 jieba 分词
    if JIEBA_AVAILABLE:
        words = jieba.lcut(text)
        # 过滤：长度>=2的中文词，或者长度>=3的数字/英文
        keywords = []
        for w in words:
            w = w.strip()
            if len(w) >= 2:
                # 中文词
                if re.match(r'[\u4e00-\u9fff]', w):
                    keywords.append(w)
                # 数字
                elif re.match(r'\d+(?:\.\d+)?', w):
                    keywords.append(w)
                # 英文/混合词
                elif len(w) >= 3:
                    keywords.append(w)
        return keywords

    # 保底方案：按长度分割
    if len(text) <= 10:
        return [text]

    # 按2-4字切分
    keywords = []
    for i in range(0, len(text), 2):
        chunk = text[i:i + 4]
        if len(chunk) >= 2:
            keywords.append(chunk)
    return keywords[:5]  # 最多5个关键词


def calculate_match_score(text: str, keywords: List[str]) -> Tuple[float, List[str]]:
    """
    计算匹配分数

    返回:
        (匹配分数, 匹配到的关键词列表)
    """
    if not keywords:
        return 0.0, []

    matched = []
    for kw in keywords:
        if kw in text:
            matched.append(kw)

    score = len(matched) / len(keywords)
    return score, matched


async def evaluate():
    print("=" * 70)
    print("招投标智能问答系统 - 准确率测试")
    print("=" * 70)

    # 加载评估集
    eval_path = Path("./data/manual_eval_set.json")
    if not eval_path.exists():
        print(f"\n❌ 评估集文件不存在: {eval_path}")
        print("\n请创建评估集文件，格式示例:")
        print('''
[
  {
    "question": "超声波流量计检测服务项目的中标人是谁",
    "expected_answer": "重庆稳信科技有限公司"
  },
  {
    "question": "招标法第三条是什么",
    "expected_answer": "必须进行招标 工程建设项目"
  }
]
        ''')
        return

    with open(eval_path, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)

    print(f"\n📋 加载评估集: {len(test_cases)} 条")

    # 初始化检索器和路由器
    print("\n🔧 初始化检索器...")
    retriever = HybridRetriever()
    router = IntentRouter()

    # 统计
    results = []
    correct = 0
    total = len(test_cases)

    # 详细结果
    detailed_results = []

    print("\n" + "=" * 70)
    print("开始测试...")
    print("=" * 70)

    for i, case in enumerate(test_cases, 1):
        question = case.get("question", "")
        expected = case.get("expected_answer", "")

        if not question:
            print(f"\n[{i}/{total}] ⚠️ 跳过: 问题为空")
            continue

        print(f"\n[{i}/{total}] 📝 问题: {question[:60]}...")

        # 1. 意图识别
        route = router.route(question)
        print(f"   🧭 路由: {route['collection']}")

        # 2. 检索
        try:
            results_list = retriever.search(
                query=question,
                collection=route["collection"],
                top_k=5
            )
        except Exception as e:
            print(f"   ❌ 检索失败: {e}")
            detailed_results.append({
                "question": question,
                "expected": expected,
                "matched": False,
                "error": str(e)
            })
            continue

        if not results_list:
            print(f"   ❌ 无检索结果")
            detailed_results.append({
                "question": question,
                "expected": expected,
                "matched": False,
                "reason": "无检索结果"
            })
            continue

        # 3. 提取关键词
        keywords = extract_keywords(expected)
        print(f"   🔑 关键词: {keywords[:5]}...")

        # 4. 检查前5条结果中是否有匹配
        found = False
        best_match_score = 0.0
        best_match_idx = -1
        matched_keywords = []

        for idx, r in enumerate(results_list[:5]):
            text = r.get("text", "")
            # 也检查 metadata 中的关键字段
            metadata = r.get("metadata", {})
            metadata_text = " ".join([
                str(metadata.get("project_name", "")),
                str(metadata.get("winner", "")),
                str(metadata.get("title", ""))
            ])

            full_text = text + " " + metadata_text

            score, matched = calculate_match_score(full_text, keywords)

            if score > best_match_score:
                best_match_score = score
                best_match_idx = idx
                matched_keywords = matched

            if score >= 0.5:  # 50% 匹配率就算正确
                found = True
                print(f"   ✅ 第{idx + 1}位命中 (匹配度: {score:.0%}, 匹配词: {matched})")
                break

        if found:
            correct += 1
        else:
            # 显示最佳匹配的信息
            if best_match_idx >= 0:
                print(f"   ❌ 未命中 (最佳匹配: 第{best_match_idx + 1}位, 匹配度: {best_match_score:.0%})")
                # 显示第一个结果的预览
                first_text = results_list[0].get("text", "")[:100]
                print(f"   📄 首位结果预览: {first_text}...")
            else:
                print(f"   ❌ 未命中 (无有效匹配)")

        # 保存详细结果
        detailed_results.append({
            "question": question,
            "expected": expected,
            "keywords": keywords,
            "matched": found,
            "best_match_score": best_match_score,
            "best_match_idx": best_match_idx,
            "matched_keywords": matched_keywords
        })

    # 计算准确率
    accuracy = correct / total * 100 if total > 0 else 0

    print("\n" + "=" * 70)
    print("📊 测试结果汇总")
    print("=" * 70)
    print(f"  总用例数: {total}")
    print(f"  正确数: {correct}")
    print(f"  准确率: {accuracy:.1f}%")
    print(f"  目标值: 92%")

    if accuracy >= 92:
        print(f"  ✅ 达标！")
    else:
        print(f"  ❌ 未达标，差 {92 - accuracy:.1f} 个百分点")

    print("=" * 70)

    # 显示失败的用例
    failed = [d for d in detailed_results if not d.get("matched", False)]
    if failed:
        print(f"\n📋 失败用例 ({len(failed)} 条):")
        print("-" * 50)
        for f in failed[:10]:  # 最多显示10条
            print(f"\n  问题: {f['question'][:50]}...")
            print(f"  期望关键词: {f.get('keywords', [])[:3]}")
            if 'best_match_score' in f:
                print(f"  最佳匹配度: {f['best_match_score']:.0%}")

    # 保存详细报告
    report_path = Path("./data/test_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "details": detailed_results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n📁 详细报告已保存: {report_path}")

    return accuracy


if __name__ == "__main__":
    asyncio.run(evaluate())