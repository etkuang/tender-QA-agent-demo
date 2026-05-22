#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
融合策略对比测试脚本 - 使用独立评估模型（客观评估）

使用方式：
    python compare_fusion.py
    python compare_fusion.py --eval-set data/eval_set_200.json --eval-model "BAAI/bge-large-zh"
"""

# ========== 必须在 import config 之前设置环境变量 ==========
import os
os.environ["MIN_FINAL_SCORE"] = "0.0"
print("📊 评估模式: MIN_FINAL_SCORE=0.0 (临时覆盖)")

import asyncio
import json
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, field

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))

from app.core.retriever import HybridRetriever
from config import settings


# ========== 评估模型管理器 ==========

class EvaluationModel:
    """评估模型 - 作为客观裁判，与检索模型分离"""

    _instance = None
    _model = None
    _model_name = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def init(self, model_name: str = "BAAI/bge-large-zh", model_type: str = "sentence_transformers"):
        self._model_name = model_name
        self._model_type = model_type

        if model_type == "sentence_transformers":
            print(f"🔄 加载评估模型 (裁判): {model_name}")
            self._model = SentenceTransformer(model_name)
        elif model_type == "openai":
            print(f"🔄 使用 OpenAI 模型作为裁判: {model_name}")
            try:
                import openai
                self._model = "openai"
            except ImportError:
                print("⚠️ 请安装 openai: pip install openai")
                self._model = SentenceTransformer("BAAI/bge-large-zh")

    def compute_similarity(self, text1: str, text2: str) -> float:
        if not text1 or not text2:
            return 0.0

        text1 = text1[:1000]
        text2 = text2[:1000]

        if self._model_type == "sentence_transformers":
            embeddings = self._model.encode([text1, text2])
        elif self._model_type == "openai":
            import openai
            embeddings = []
            for text in [text1, text2]:
                response = openai.Embedding.create(
                    model=self._model_name,
                    input=text
                )
                embeddings.append(response['data'][0]['embedding'])
            embeddings = np.array(embeddings)
        else:
            return 0.0

        similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return float(similarity)


# ========== 数据结构 ==========

@dataclass
class QueryEvalResult:
    qid: str
    category: str
    question: str
    answer: str
    source: str
    retrieved_docs: List[Dict]
    total_relevant: int = 0
    relevant_ranks: List[int] = field(default_factory=list)
    relevance_scores: Dict[int, float] = field(default_factory=dict)
    best_rank: Optional[int] = None
    max_relevance: float = 0.0
    precision_at_1: bool = False
    precision_at_3: bool = False
    precision_at_5: bool = False
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    reciprocal_rank: float = 0.0


@dataclass
class StrategyResult:
    """单个策略的评估结果"""
    strategy: str
    retrieval_model: str
    evaluation_model: str
    total_questions: int
    valid_questions: int
    weighted_precision_1: float = 0.0
    weighted_precision_3: float = 0.0
    weighted_precision_5: float = 0.0
    weighted_recall_1: float = 0.0
    weighted_recall_3: float = 0.0
    weighted_recall_5: float = 0.0
    weighted_mrr: float = 0.0
    category_results: Dict = field(default_factory=dict)


# ========== 评估器 ==========

class RecallEvaluator:
    """召回率评估器 - 使用独立的评估模型"""

    def __init__(self, eval_model_name: str = "BAAI/bge-large-zh",
                 eval_model_type: str = "sentence_transformers",
                 similarity_threshold: float = 0.65):

        self.retrieval_model = settings.embedding_model
        self.similarity_threshold = similarity_threshold

        # 初始化独立的评估模型
        self.eval_model = EvaluationModel()
        self.eval_model.init(eval_model_name, eval_model_type)
        self.evaluation_model_name = eval_model_name

    def _extract_text_from_doc(self, doc: Dict) -> str:
        text = doc.get("text", "")
        if not text:
            data = doc.get("data", {})
            text = data.get("text", "") or data.get("content", "") or ""
        return text

    async def evaluate_single_query(self, retriever, item: Dict, top_k: int = 10) -> QueryEvalResult:
        qid = item["id"]
        category = item.get("category", "unknown")
        question = item["question"]
        answer = item["answer"]
        source = item.get("source", "")

        try:
            results = await asyncio.to_thread(
                retriever.search,
                query=question,
                collection="regulations",
                top_k=top_k
            )
        except Exception as e:
            print(f"      ⚠️ 检索失败: {e}")
            results = []

        if not results:
            return QueryEvalResult(
                qid=qid, category=category, question=question,
                answer=answer, source=source, retrieved_docs=[]
            )

        relevant_ranks = []
        relevance_scores = {}
        max_relevance = 0.0

        for rank, doc in enumerate(results, 1):
            doc_text = self._extract_text_from_doc(doc)
            similarity = self.eval_model.compute_similarity(doc_text, answer)
            relevance_scores[rank] = similarity
            max_relevance = max(max_relevance, similarity)

            if similarity >= self.similarity_threshold:
                relevant_ranks.append(rank)

        total_relevant = len(relevant_ranks)

        precision_at_1 = 1 in relevant_ranks
        precision_at_3 = any(r <= 3 for r in relevant_ranks)
        precision_at_5 = any(r <= 5 for r in relevant_ranks)

        recall_at_1 = sum(1 for r in relevant_ranks if r <= 1) / total_relevant if total_relevant > 0 else 0
        recall_at_3 = sum(1 for r in relevant_ranks if r <= 3) / total_relevant if total_relevant > 0 else 0
        recall_at_5 = sum(1 for r in relevant_ranks if r <= 5) / total_relevant if total_relevant > 0 else 0

        best_rank = min(relevant_ranks) if relevant_ranks else None
        reciprocal_rank = 1.0 / best_rank if best_rank else 0.0

        return QueryEvalResult(
            qid=qid, category=category, question=question,
            answer=answer, source=source, retrieved_docs=results[:5],
            total_relevant=total_relevant, relevant_ranks=relevant_ranks,
            relevance_scores=relevance_scores, best_rank=best_rank,
            max_relevance=max_relevance,
            precision_at_1=precision_at_1, precision_at_3=precision_at_3, precision_at_5=precision_at_5,
            recall_at_1=recall_at_1, recall_at_3=recall_at_3, recall_at_5=recall_at_5,
            reciprocal_rank=reciprocal_rank
        )

    async def evaluate_strategy(self, strategy: str, eval_set_path: Path, top_k: int = 10) -> StrategyResult:
        """评估单个融合策略"""
        print(f"\n{'=' * 60}")
        print(f"📊 开始评估 - 融合策略: {strategy.upper()}")
        print(f"{'=' * 60}")

        # 设置融合策略
        old_strategy = settings.fusion_strategy
        settings.fusion_strategy = strategy

        try:
            retriever = HybridRetriever()

            # 加载评估集
            with open(eval_set_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

            if isinstance(raw_data, dict) and "questions" in raw_data:
                questions = raw_data["questions"]
            else:
                questions = raw_data

            print(f"📚 加载评估集: {eval_set_path}")
            print(f"   问题总数: {len(questions)}")

            all_results = []
            total = len(questions)
            failed_count = 0

            for i, item in enumerate(questions, 1):
                print(f"   [{i}/{total}] {item['id']}: {item['question'][:50]}...")
                result = await self.evaluate_single_query(retriever, item, top_k)
                if not result.retrieved_docs:
                    failed_count += 1
                all_results.append(result)

            if failed_count > 0:
                print(f"   ⚠️ 检索失败: {failed_count} 条查询返回空结果")

            # 按类别汇总
            categories = set(r.category for r in all_results)
            category_results = {}

            for category in categories:
                cat_results = [r for r in all_results if r.category == category]
                total_count = len(cat_results)
                valid_count = len([r for r in cat_results if r.retrieved_docs])

                precision_1 = sum(1 for r in cat_results if r.precision_at_1) / total_count * 100
                precision_3 = sum(1 for r in cat_results if r.precision_at_3) / total_count * 100
                precision_5 = sum(1 for r in cat_results if r.precision_at_5) / total_count * 100
                recall_1 = sum(r.recall_at_1 for r in cat_results) / total_count * 100
                recall_3 = sum(r.recall_at_3 for r in cat_results) / total_count * 100
                recall_5 = sum(r.recall_at_5 for r in cat_results) / total_count * 100
                mrr = sum(r.reciprocal_rank for r in cat_results) / total_count * 100

                category_results[category] = {
                    "total": total_count,
                    "valid": valid_count,
                    "precision_1": round(precision_1, 2),
                    "precision_3": round(precision_3, 2),
                    "precision_5": round(precision_5, 2),
                    "recall_1": round(recall_1, 2),
                    "recall_3": round(recall_3, 2),
                    "recall_5": round(recall_5, 2),
                    "mrr": round(mrr, 2)
                }

            # 加权平均
            total_questions = sum(c["total"] for c in category_results.values())

            if total_questions > 0:
                weighted_precision_1 = sum(
                    c["precision_1"] * c["total"] for c in category_results.values()) / total_questions
                weighted_precision_3 = sum(
                    c["precision_3"] * c["total"] for c in category_results.values()) / total_questions
                weighted_precision_5 = sum(
                    c["precision_5"] * c["total"] for c in category_results.values()) / total_questions
                weighted_recall_1 = sum(c["recall_1"] * c["total"] for c in category_results.values()) / total_questions
                weighted_recall_3 = sum(c["recall_3"] * c["total"] for c in category_results.values()) / total_questions
                weighted_recall_5 = sum(c["recall_5"] * c["total"] for c in category_results.values()) / total_questions
                weighted_mrr = sum(c["mrr"] * c["total"] for c in category_results.values()) / total_questions
            else:
                weighted_precision_1 = weighted_precision_3 = weighted_precision_5 = 0
                weighted_recall_1 = weighted_recall_3 = weighted_recall_5 = 0
                weighted_mrr = 0

            print(f"\n   ✅ {strategy.upper()} 结果:")
            print(f"      Precision@1: {weighted_precision_1:.1f}%")
            print(f"      Recall@1:    {weighted_recall_1:.1f}%")
            print(f"      MRR:         {weighted_mrr:.1f}%")
            print(f"      有效查询:    {total_questions - failed_count}/{total_questions}")

            return StrategyResult(
                strategy=strategy,
                retrieval_model=self.retrieval_model,
                evaluation_model=self.evaluation_model_name,
                total_questions=len(questions),
                valid_questions=total_questions - failed_count,
                weighted_precision_1=weighted_precision_1,
                weighted_precision_3=weighted_precision_3,
                weighted_precision_5=weighted_precision_5,
                weighted_recall_1=weighted_recall_1,
                weighted_recall_3=weighted_recall_3,
                weighted_recall_5=weighted_recall_5,
                weighted_mrr=weighted_mrr,
                category_results=category_results
            )

        finally:
            settings.fusion_strategy = old_strategy


# ========== 对比器 ==========

class FusionComparator:
    """融合策略对比器"""

    def __init__(self, eval_set_path: Path, eval_model_name: str = "BAAI/bge-large-zh",
                 eval_model_type: str = "sentence_transformers", similarity_threshold: float = 0.65):
        self.eval_set_path = eval_set_path
        self.similarity_threshold = similarity_threshold
        self.eval_model_name = eval_model_name
        self.eval_model_type = eval_model_type
        self.results: Dict[str, StrategyResult] = {}

    async def compare(self):
        """执行对比测试"""
        print("\n" + "=" * 80)
        print("🔍 融合策略对比测试 - 客观评估模式")
        print("=" * 80)
        print(f"\n📋 评估集: {self.eval_set_path}")
        print(f"   相似度阈值: {self.similarity_threshold}")
        print(f"\n⚙️ 模型配置:")
        print(f"   检索模型 (系统使用): {settings.embedding_model}")
        print(f"   评估模型 (裁判):     {self.eval_model_name}")
        if settings.embedding_model != self.eval_model_name:
            print(f"   ✅ 两个模型不同，评估结果客观！")
        else:
            print(f"   ⚠️  两个模型相同，评估结果可能虚高！")

        evaluator = RecallEvaluator(
            eval_model_name=self.eval_model_name,
            eval_model_type=self.eval_model_type,
            similarity_threshold=self.similarity_threshold
        )

        strategies = ["weighted", "rrf"]

        for strategy in strategies:
            result = await evaluator.evaluate_strategy(strategy, self.eval_set_path)
            self.results[strategy] = result

        self.print_comparison()
        self.save_comparison_report()

    def print_comparison(self):
        """打印对比报告"""
        print("\n" + "=" * 110)
        print("📊 融合策略对比报告 - 客观评估 (裁判模型独立)")
        print("=" * 110)

        if "weighted" not in self.results or "rrf" not in self.results:
            print("❌ 缺少对比数据")
            return

        weighted = self.results["weighted"]
        rrf = self.results["rrf"]

        print(f"\n⚙️ 评估配置:")
        print(f"   检索模型: {weighted.retrieval_model}")
        print(f"   评估模型: {weighted.evaluation_model}")
        print(f"\n📊 有效查询统计:")
        print(f"   Weighted: {weighted.valid_questions}/{weighted.total_questions} 条有效")
        print(f"   RRF:      {rrf.valid_questions}/{rrf.total_questions} 条有效")

        if weighted.valid_questions < weighted.total_questions:
            print(f"   ⚠️ 注意: Weighted 策略有 {weighted.total_questions - weighted.valid_questions} 条查询失败")
            print(f"      失败查询已计为 0 分，确保公平对比")

        print("\n" + "-" * 110)
        print(f"{'指标':<20} {'Weighted':>15} {'RRF':>15} {'差异':>15} {'优胜者':>12}")
        print("-" * 110)

        metrics = [
            ("Precision@1 (%)", weighted.weighted_precision_1, rrf.weighted_precision_1),
            ("Precision@3 (%)", weighted.weighted_precision_3, rrf.weighted_precision_3),
            ("Precision@5 (%)", weighted.weighted_precision_5, rrf.weighted_precision_5),
            ("Recall@1 (%)", weighted.weighted_recall_1, rrf.weighted_recall_1),
            ("Recall@3 (%)", weighted.weighted_recall_3, rrf.weighted_recall_3),
            ("Recall@5 (%)", weighted.weighted_recall_5, rrf.weighted_recall_5),
            ("MRR (%)", weighted.weighted_mrr, rrf.weighted_mrr),
        ]

        for name, w_val, r_val in metrics:
            diff = w_val - r_val
            if diff > 1:
                winner = "Weighted 🏆"
            elif diff < -1:
                winner = "RRF 🏆"
            else:
                winner = "持平"
            print(f"{name:<20} {w_val:>14.1f}% {r_val:>14.1f}% {diff:>14.1f}% {winner:>12}")

        print("-" * 110)

        # 分类别对比
        print("\n📈 分类别 Precision@1 对比:")
        print("-" * 80)
        print(f"{'类别':<12} {'Weighted':>12} {'RRF':>12} {'差异':>12} {'优胜者':>10}")
        print("-" * 80)

        category_order = ["fact", "reasoning", "comparison", "application", "extended"]
        for cat in category_order:
            if cat in weighted.category_results and cat in rrf.category_results:
                w_acc = weighted.category_results[cat]["precision_1"]
                r_acc = rrf.category_results[cat]["precision_1"]
                diff = w_acc - r_acc
                winner = "Weighted" if diff > 0 else ("RRF" if diff < 0 else "持平")
                print(f"{cat:<12} {w_acc:>11.1f}% {r_acc:>11.1f}% {diff:>11.1f}% {winner:>10}")

        print("-" * 80)

        print("\n📈 分类别 Recall@1 对比:")
        print("-" * 80)
        print(f"{'类别':<12} {'Weighted':>12} {'RRF':>12} {'差异':>12} {'优胜者':>10}")
        print("-" * 80)

        for cat in category_order:
            if cat in weighted.category_results and cat in rrf.category_results:
                w_rec = weighted.category_results[cat]["recall_1"]
                r_rec = rrf.category_results[cat]["recall_1"]
                diff = w_rec - r_rec
                winner = "Weighted" if diff > 0 else ("RRF" if diff < 0 else "持平")
                print(f"{cat:<12} {w_rec:>11.1f}% {r_rec:>11.1f}% {diff:>11.1f}% {winner:>10}")

        print("-" * 80)

        # 可视化
        print("\n📊 可视化对比:")
        print("-" * 60)

        for name, w_val, r_val in [
            ("P@1", weighted.weighted_precision_1, rrf.weighted_precision_1),
            ("R@1", weighted.weighted_recall_1, rrf.weighted_recall_1),
        ]:
            w_bar = "█" * int(w_val / 5)
            r_bar = "█" * int(r_val / 5)
            print(f"   {name}   Weighted: {w_val:5.1f}%  {w_bar}")
            print(f"        RRF:     {r_val:5.1f}%  {r_bar}")
            print()

        # 总结
        print("\n💡 总结建议:")
        if weighted.weighted_recall_1 > rrf.weighted_recall_1:
            print("   ✅ Weighted 融合策略在召回率上表现更好")
            print(f"      比 RRF 高出 {weighted.weighted_recall_1 - rrf.weighted_recall_1:.1f}%")
        elif rrf.weighted_recall_1 > weighted.weighted_recall_1:
            print("   ✅ RRF 融合策略在召回率上表现更好")
            print(f"      比 Weighted 高出 {rrf.weighted_recall_1 - weighted.weighted_recall_1:.1f}%")
        else:
            print("   🤝 两个策略表现相近")

        print("=" * 110)

    def save_comparison_report(self):
        """保存对比报告"""
        output_dir = Path("./data/eval_results")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        comparison = {
            "timestamp": timestamp,
            "eval_set_path": str(self.eval_set_path),
            "similarity_threshold": self.similarity_threshold,
            "retrieval_model": settings.embedding_model,
            "evaluation_model": self.eval_model_name,
            "note": "评估模式: MIN_FINAL_SCORE=0.0，失败查询计为0分",
            "results": {}
        }

        for strategy, result in self.results.items():
            comparison["results"][strategy] = {
                "total_questions": result.total_questions,
                "valid_questions": result.valid_questions,
                "precision_1": round(result.weighted_precision_1, 2),
                "precision_3": round(result.weighted_precision_3, 2),
                "precision_5": round(result.weighted_precision_5, 2),
                "recall_1": round(result.weighted_recall_1, 2),
                "recall_3": round(result.weighted_recall_3, 2),
                "recall_5": round(result.weighted_recall_5, 2),
                "mrr": round(result.weighted_mrr, 2),
            }

        report_path = output_dir / f"fusion_comparison_{timestamp}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)

        print(f"\n💾 对比报告已保存: {report_path}")


async def main():
    parser = argparse.ArgumentParser(description="融合策略对比测试 - 客观评估")
    parser.add_argument("--eval-set", type=str, default="./data/eval_set_200.json",
                        help="评估集文件路径")
    parser.add_argument("--eval-model", type=str, default=r"C:\Users\ian\.cache\huggingface\hub\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181",
                        help="评估模型（裁判），推荐比检索模型更强")
    parser.add_argument("--eval-type", type=str, default="sentence_transformers",
                        choices=["sentence_transformers", "openai"],
                        help="评估模型类型")
    parser.add_argument("--threshold", type=float, default=0.65,
                        help="语义相似度阈值")

    args = parser.parse_args()

    eval_set_path = Path(args.eval_set)
    if not eval_set_path.exists():
        print(f"❌ 评估集不存在: {eval_set_path}")
        return

    comparator = FusionComparator(
        eval_set_path=eval_set_path,
        eval_model_name=args.eval_model,
        eval_model_type=args.eval_type,
        similarity_threshold=args.threshold
    )
    await comparator.compare()


if __name__ == "__main__":
    asyncio.run(main())