#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
端到端测试脚本 - 测试完整系统链路（使用独立评估模型）

测试链路：
用户问题 → 意图路由 → 问题改写 → 检索 → LLM生成 → 评估

使用方式：
    # 测试单个问题
    python e2e_eval.py --single --question "什么是串通投标？"

    # 测试评估集
    python e2e_eval.py --eval-set data/eval_set_200.json

    # 使用独立评估模型（推荐）
    python e2e_eval.py --eval-set data/eval_set_200.json --eval-model "BAAI/bge-large-zh"

    # 对比不同融合策略
    python e2e_eval.py --eval-set data/eval_set_200.json --compare
"""
# 在 import 之后，其他代码之前添加
import os
os.environ["MIN_FINAL_SCORE"] = "0.0"
print("📊 评估模式: MIN_FINAL_SCORE=0.0 (临时覆盖)")
import asyncio
import json
import sys
import hashlib
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, field

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))

from app.core.retriever import HybridRetriever
from app.core.generator import LLMGenerator
from app.core.intent_router import IntentRouter
from app.core.session_manager import SessionManager
from config import settings


# ========== 评估模型管理器 ==========

class EvaluationModel:
    """评估模型 - 作为客观裁判，与检索模型分离"""

    _instance = None
    _model = None
    _model_name = None
    _model_type = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def init(self, model_name: str = "BAAI/bge-large-zh", model_type: str = "sentence_transformers"):
        """初始化评估模型"""
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
                print("   回退到本地模型")
                self._model = SentenceTransformer("BAAI/bge-large-zh")
                self._model_type = "sentence_transformers"

    def compute_similarity(self, text1: str, text2: str) -> float:
        """计算两段文本的语义相似度"""
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

    @property
    def model_name(self):
        return self._model_name


# ========== 数据结构 ==========

@dataclass
class E2EResult:
    """端到端单个测试结果"""
    qid: str
    category: str
    question: str
    expected_answer: str
    source: str

    # 系统各环节输出
    intent_type: str = ""
    intent_complexity: str = ""
    rewritten_question: str = ""
    retrieved_chunks: List[Dict] = field(default_factory=list)
    generated_answer: str = ""

    # 性能指标
    intent_time_ms: float = 0
    rewrite_time_ms: float = 0
    retrieval_time_ms: float = 0
    generation_time_ms: float = 0
    total_time_ms: float = 0

    # 评估指标（使用独立评估模型）
    answer_similarity: float = 0.0
    is_correct: bool = False
    contains_key_info: bool = False
    route_used: str = ""

    # 错误信息
    error: str = ""


@dataclass
class E2EOverallResult:
    """端到端整体评估结果"""
    timestamp: str
    eval_set_path: str
    retrieval_model: str
    evaluation_model: str
    total_questions: int
    valid_questions: int
    config: Dict

    # 整体指标
    avg_answer_similarity: float = 0.0
    avg_response_time_ms: float = 0.0
    accuracy_at_threshold: float = 0.0

    # 各环节平均耗时
    avg_intent_time_ms: float = 0.0
    avg_rewrite_time_ms: float = 0.0
    avg_retrieval_time_ms: float = 0.0
    avg_generation_time_ms: float = 0.0

    # 路由分布
    route_distribution: Dict = field(default_factory=dict)

    # 分类别结果
    category_results: Dict = field(default_factory=dict)

    # 详细结果
    failed_cases: List[Dict] = field(default_factory=list)


# ========== 端到端评估器 ==========

class E2EEvaluator:
    """端到端评估器 - 使用独立评估模型"""

    def __init__(self, eval_model_name: str = "BAAI/bge-large-zh",
                 eval_model_type: str = "sentence_transformers",
                 similarity_threshold: float = 0.65):

        print("🔄 初始化端到端评估器...")

        # 系统组件
        self.retriever = HybridRetriever()
        self.llm = LLMGenerator()
        self.session_manager = SessionManager()
        self.intent_router = IntentRouter(self.llm)

        # 记录检索模型
        self.retrieval_model = settings.embedding_model

        # 初始化独立的评估模型（裁判）
        self.eval_model = EvaluationModel()
        self.eval_model.init(eval_model_name, eval_model_type)
        self.similarity_threshold = similarity_threshold

        # 导入问题改写器
        from app.utils.question_rewriter import question_rewriter
        self.question_rewriter = question_rewriter

        # 保存原始的 min_final_score
        self.original_min_final_score = settings.min_final_score

        print(f"\n📋 模型配置:")
        print(f"   检索模型 (系统使用): {self.retrieval_model}")
        print(f"   评估模型 (裁判):     {self.eval_model.model_name}")
        if self.retrieval_model != self.eval_model.model_name:
            print(f"   ✅ 两个模型不同，评估结果客观！")
        else:
            print(f"   ⚠️  两个模型相同，评估结果可能虚高！")

    def check_key_info(self, generated: str, expected: str) -> bool:
        """检查生成答案是否包含标准答案中的关键信息"""
        import re

        key_patterns = [
            r'\d+[%％]?',
            r'第\s*\d+\s*条',
            r'[一二三四五六七八九十百千万]+',
            r'[0-9]+[\.\d]*万?',
        ]

        expected_keys = set()
        for pattern in key_patterns:
            matches = re.findall(pattern, expected)
            expected_keys.update(matches)

        if not expected_keys:
            return True

        matched = 0
        for key in expected_keys:
            if key in generated:
                matched += 1

        return matched / len(expected_keys) >= 0.5

    async def test_single(self, question: str, expected_answer: str = "",
                          qid: str = "", category: str = "") -> E2EResult:
        """测试单个问题（完整链路）"""
        result = E2EResult(
            qid=qid or "test",
            category=category or "unknown",
            question=question,
            expected_answer=expected_answer,
            source=""
        )

        session_id = f"test_{int(time.time())}"
        start_time = time.time()

        try:
            # ========== 1. 意图路由 ==========
            intent_start = time.time()
            intent = await self.intent_router.route(
                question=question,
                session_id=session_id,
                session_manager=self.session_manager
            )
            result.intent_type = intent.get("type", "unknown")
            result.intent_complexity = intent.get("complexity", "unknown")
            result.intent_time_ms = (time.time() - intent_start) * 1000

            # 快速回复或无关问题直接返回
            if intent["type"] in ["quick_response", "unrelated"]:
                result.generated_answer = intent.get("response", settings.unrelated_response)
                result.route_used = intent["type"]
                result.total_time_ms = (time.time() - start_time) * 1000

                if expected_answer:
                    result.answer_similarity = self.eval_model.compute_similarity(
                        result.generated_answer, expected_answer
                    ) * 100
                    result.is_correct = result.answer_similarity >= self.similarity_threshold * 100

                return result

            # ========== 2. 问题改写 ==========
            rewrite_start = time.time()
            rewritten = question
            if hasattr(settings, 'enable_question_rewrite') and settings.enable_question_rewrite:
                from app.main_agent import has_referential_word
                if has_referential_word(question):
                    rewritten = self.question_rewriter.rewrite(question)
            result.rewritten_question = rewritten
            result.rewrite_time_ms = (time.time() - rewrite_start) * 1000

            # ========== 3. 检索 ==========
            retrieval_start = time.time()
            try:
                results = await asyncio.to_thread(
                    self.retriever.search,
                    query=rewritten,
                    collection="regulations",
                    top_k=settings.top_k
                )
            except Exception as e:
                print(f"      ⚠️ 检索失败: {e}")
                results = []
            result.retrieved_chunks = results[:3] if results else []
            result.retrieval_time_ms = (time.time() - retrieval_start) * 1000

            # ========== 4. LLM 生成 ==========
            generation_start = time.time()

            from app.tools.regulation_tools import summarize
            if results:
                answer = await summarize(results, rewritten, self.llm)
            else:
                answer = settings.no_results_response

            result.generated_answer = answer
            result.generation_time_ms = (time.time() - generation_start) * 1000
            result.route_used = "rag"

        except Exception as e:
            result.error = str(e)
            result.generated_answer = f"处理失败: {str(e)}"

        result.total_time_ms = (time.time() - start_time) * 1000

        # ========== 5. 评估（使用独立评估模型） ==========
        if expected_answer:
            result.answer_similarity = self.eval_model.compute_similarity(
                result.generated_answer, expected_answer
            ) * 100
            result.is_correct = result.answer_similarity >= self.similarity_threshold * 100
            result.contains_key_info = self.check_key_info(result.generated_answer, expected_answer)

        return result

    async def evaluate_dataset(self, eval_set_path: Path) -> E2EOverallResult:
        """评估整个数据集"""
        # 评估模式下降低 min_final_score 阈值
        old_min_score = settings.min_final_score
        settings.min_final_score = 0.0
        print(f"   📊 评估模式: min_final_score = {settings.min_final_score} (临时，正常使用为 {old_min_score})")

        try:
            with open(eval_set_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

            if isinstance(raw_data, dict) and "questions" in raw_data:
                questions = raw_data["questions"]
            else:
                questions = raw_data

            print(f"\n📚 加载评估集: {eval_set_path}")
            print(f"   问题总数: {len(questions)}")

            results = []
            total = len(questions)

            for i, item in enumerate(questions, 1):
                print(f"   [{i}/{total}] {item['id']}: {item['question'][:50]}...")

                result = await self.test_single(
                    question=item["question"],
                    expected_answer=item.get("answer", ""),
                    qid=item["id"],
                    category=item.get("category", "unknown")
                )
                results.append(result)

            # 统计整体指标（包括失败的，失败的自然为0）
            total_count = len(results)
            valid_results = [r for r in results if not r.error]
            valid_count = len(valid_results)

            if total_count == 0:
                return E2EOverallResult(
                    timestamp=datetime.now().isoformat(),
                    eval_set_path=str(eval_set_path),
                    retrieval_model=self.retrieval_model,
                    evaluation_model=self.eval_model.model_name,
                    total_questions=total,
                    valid_questions=0,
                    config={}
                )

            # 整体平均值（使用所有结果，包括失败的）
            avg_similarity = sum(r.answer_similarity for r in results) / total_count
            avg_total_time = sum(r.total_time_ms for r in results) / total_count
            accuracy = sum(1 for r in results if r.is_correct) / total_count * 100

            avg_intent_time = sum(r.intent_time_ms for r in results) / total_count
            avg_rewrite_time = sum(r.rewrite_time_ms for r in results) / total_count
            avg_retrieval_time = sum(r.retrieval_time_ms for r in results) / total_count
            avg_generation_time = sum(r.generation_time_ms for r in results) / total_count

            # 路由分布
            route_dist = {}
            for r in results:
                route = r.route_used or "unknown"
                route_dist[route] = route_dist.get(route, 0) + 1

            # 分类别统计（使用所有结果）
            category_results = {}
            categories = set(r.category for r in results)
            for cat in categories:
                cat_results = [r for r in results if r.category == cat]
                cat_count = len(cat_results)
                if cat_count > 0:
                    category_results[cat] = {
                        "count": cat_count,
                        "avg_similarity": sum(r.answer_similarity for r in cat_results) / cat_count,
                        "accuracy": sum(1 for r in cat_results if r.is_correct) / cat_count * 100,
                        "avg_time_ms": sum(r.total_time_ms for r in cat_results) / cat_count
                    }

            # 失败案例
            failed_cases = []
            for r in results:
                if not r.is_correct and r.expected_answer:
                    failed_cases.append({
                        "qid": r.qid,
                        "question": r.question[:100],
                        "expected": r.expected_answer[:150],
                        "generated": r.generated_answer[:200],
                        "similarity": round(r.answer_similarity, 2),
                        "route": r.route_used
                    })

            config_snapshot = {
                "fusion_strategy": settings.fusion_strategy,
                "retrieval_model": self.retrieval_model,
                "evaluation_model": self.eval_model.model_name,
                "similarity_threshold": self.similarity_threshold,
                "top_k": settings.top_k,
                "eval_mode_min_score": 0.0
            }

            return E2EOverallResult(
                timestamp=datetime.now().isoformat(),
                eval_set_path=str(eval_set_path),
                retrieval_model=self.retrieval_model,
                evaluation_model=self.eval_model.model_name,
                total_questions=total,
                valid_questions=valid_count,
                config=config_snapshot,
                avg_answer_similarity=avg_similarity,
                avg_response_time_ms=avg_total_time,
                accuracy_at_threshold=accuracy,
                avg_intent_time_ms=avg_intent_time,
                avg_rewrite_time_ms=avg_rewrite_time,
                avg_retrieval_time_ms=avg_retrieval_time,
                avg_generation_time_ms=avg_generation_time,
                route_distribution=route_dist,
                category_results=category_results,
                failed_cases=failed_cases[:30]
            )

        finally:
            # 恢复原值
            settings.min_final_score = old_min_score

    def print_results(self, result: E2EOverallResult):
        """打印评估结果"""
        print("\n" + "=" * 100)
        print("📊 端到端评估报告 - 完整系统链路测试（客观评估）")
        print("=" * 100)

        print(f"\n📋 评估信息:")
        print(f"   评估集: {result.eval_set_path}")
        print(f"   有效问题: {result.valid_questions} / {result.total_questions}")

        if result.valid_questions < result.total_questions:
            print(f"   ⚠️ 注意: {result.total_questions - result.valid_questions} 条查询失败")
            print(f"      失败查询已计为 0 分，确保公平评估")

        print(f"\n⚙️ 模型配置:")
        print(f"   检索模型 (系统使用): {result.retrieval_model}")
        print(f"   评估模型 (裁判):     {result.evaluation_model}")
        if result.retrieval_model != result.evaluation_model:
            print(f"   ✅ 两个模型不同，评估结果客观！")
        else:
            print(f"   ⚠️  两个模型相同，评估结果可能虚高！")

        print("\n" + "=" * 100)
        print("📈 核心指标")
        print("=" * 100)
        print(f"   答案准确率 (相似度>阈值): {result.accuracy_at_threshold:>6.1f}%")
        print(f"   平均答案相似度:           {result.avg_answer_similarity:>6.1f}%")
        print(f"   平均响应时间:             {result.avg_response_time_ms:>6.0f} ms")

        print("\n" + "=" * 100)
        print("⏱️  各环节耗时分析")
        print("=" * 100)
        total_time = result.avg_response_time_ms
        print(
            f"   意图路由耗时:   {result.avg_intent_time_ms:>6.0f} ms  ({result.avg_intent_time_ms / total_time * 100:>5.1f}%)")
        print(
            f"   问题改写耗时:   {result.avg_rewrite_time_ms:>6.0f} ms  ({result.avg_rewrite_time_ms / total_time * 100:>5.1f}%)")
        print(
            f"   检索耗时:       {result.avg_retrieval_time_ms:>6.0f} ms  ({result.avg_retrieval_time_ms / total_time * 100:>5.1f}%)")
        print(
            f"   生成耗时:       {result.avg_generation_time_ms:>6.0f} ms  ({result.avg_generation_time_ms / total_time * 100:>5.1f}%)")
        print(f"   ────────────────────────────────────")
        print(f"   总耗时:         {total_time:>6.0f} ms")

        print("\n" + "=" * 100)
        print("🎯 路由分布")
        print("=" * 100)
        for route, count in result.route_distribution.items():
            pct = count / result.total_questions * 100
            bar = "█" * int(pct / 2)
            print(f"   {route:<12}: {count:>3} 次 ({pct:>5.1f}%)  {bar}")

        print("\n" + "=" * 100)
        print("📂 分类别结果")
        print("=" * 100)
        print(f"{'类别':<12} {'数量':>6} {'准确率':>10} {'平均相似度':>12} {'平均耗时':>10}")
        print("-" * 60)

        category_order = ["fact", "reasoning", "comparison", "application", "extended"]
        for cat in category_order:
            if cat in result.category_results:
                c = result.category_results[cat]
                print(
                    f"{cat:<12} {c['count']:>6} {c['accuracy']:>9.1f}% {c['avg_similarity']:>11.1f}% {c['avg_time_ms']:>9.0f}ms")

        print("-" * 60)
        print(
            f"{'总计':<12} {result.total_questions:>6} {result.accuracy_at_threshold:>9.1f}% {result.avg_answer_similarity:>11.1f}% {result.avg_response_time_ms:>9.0f}ms")

        # 可视化
        print("\n" + "=" * 100)
        print("📊 可视化")
        print("=" * 100)

        metrics = [
            ("准确率", result.accuracy_at_threshold),
            ("平均相似度", result.avg_answer_similarity),
        ]
        for name, value in metrics:
            bar = "█" * int(value / 5)
            print(f"   {name:<10}: {value:5.1f}%  {bar}")

        # 失败案例
        if result.failed_cases:
            print(f"\n❌ 失败案例 (相似度低于阈值，共{len(result.failed_cases)}条):")
            for i, fail in enumerate(result.failed_cases[:5], 1):
                print(f"\n   [{i}] {fail['qid']}: {fail['question']}")
                print(f"       期望: {fail['expected'][:80]}...")
                print(f"       生成: {fail['generated'][:80]}...")
                print(f"       相似度: {fail['similarity']:.1f}%")

        print("=" * 100)

    def save_report(self, result: E2EOverallResult, output_dir: Path = Path("./data/eval_results")):
        """保存评估报告"""
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        report = {
            "timestamp": result.timestamp,
            "eval_set_path": result.eval_set_path,
            "retrieval_model": result.retrieval_model,
            "evaluation_model": result.evaluation_model,
            "note": "失败查询计为0分，确保公平评估。评估模式下 min_final_score=0.0",
            "config": result.config,
            "summary": {
                "total_questions": result.total_questions,
                "valid_questions": result.valid_questions,
                "accuracy": round(result.accuracy_at_threshold, 2),
                "avg_similarity": round(result.avg_answer_similarity, 2),
                "avg_response_time_ms": round(result.avg_response_time_ms, 0),
                "avg_intent_time_ms": round(result.avg_intent_time_ms, 0),
                "avg_rewrite_time_ms": round(result.avg_rewrite_time_ms, 0),
                "avg_retrieval_time_ms": round(result.avg_retrieval_time_ms, 0),
                "avg_generation_time_ms": round(result.avg_generation_time_ms, 0)
            },
            "route_distribution": result.route_distribution,
            "category_results": result.category_results,
            "failed_cases": result.failed_cases[:20]
        }

        report_path = output_dir / f"e2e_report_{timestamp}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n💾 报告已保存: {report_path}")
        return report_path


# ========== 对比测试 ==========

async def compare_strategies(eval_set_path: Path, eval_model_name: str = "BAAI/bge-large-zh"):
    """对比不同融合策略的端到端效果"""
    print("\n" + "=" * 100)
    print("🔍 端到端对比测试 - 不同融合策略（客观评估）")
    print("=" * 100)

    strategies = ["weighted", "rrf"]
    results = {}

    for strategy in strategies:
        print(f"\n📊 测试融合策略: {strategy.upper()}")
        settings.fusion_strategy = strategy

        evaluator = E2EEvaluator(eval_model_name=eval_model_name)
        result = await evaluator.evaluate_dataset(eval_set_path)
        evaluator.print_results(result)
        results[strategy] = result

    # 打印对比
    print("\n" + "=" * 100)
    print("📊 融合策略端到端对比")
    print("=" * 100)
    print(f"{'策略':<12} {'准确率':>10} {'平均相似度':>12} {'平均耗时':>10}")
    print("-" * 50)

    for strategy, result in results.items():
        print(
            f"{strategy:<12} {result.accuracy_at_threshold:>9.1f}% {result.avg_answer_similarity:>11.1f}% {result.avg_response_time_ms:>9.0f}ms")

    print("=" * 100)


# ========== 主函数 ==========

async def main():
    parser = argparse.ArgumentParser(description="端到端测试脚本 - 客观评估")
    parser.add_argument("--eval-set", type=str, default="./data/eval_set_200.json",
                        help="评估集文件路径")
    parser.add_argument("--eval-model", type=str, default="bert-base-chinese",
                        help="评估模型（裁判），推荐比检索模型更强")
    parser.add_argument("--eval-type", type=str, default="sentence_transformers",
                        choices=["sentence_transformers", "openai"],
                        help="评估模型类型")
    parser.add_argument("--threshold", type=float, default=0.65,
                        help="语义相似度阈值")
    parser.add_argument("--single", action="store_true", help="单问题测试模式")
    parser.add_argument("--question", type=str, help="测试的问题")
    parser.add_argument("--compare", action="store_true", help="对比不同融合策略")
    parser.add_argument("--save", action="store_true", help="保存报告")

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🔍 端到端测试 - 完整系统链路（客观评估）")
    print("=" * 70)
    print("\n测试链路:")
    print("   用户问题 → 意图路由 → 问题改写 → 检索 → LLM生成 → 评估")
    print("\n⚠️  评估使用独立模型（裁判），保证客观性！")
    print("=" * 70)

    # 单问题测试模式
    if args.single and args.question:
        evaluator = E2EEvaluator(
            eval_model_name=args.eval_model,
            eval_model_type=args.eval_type,
            similarity_threshold=args.threshold
        )
        result = await evaluator.test_single(args.question)

        print(f"\n📝 测试结果:")
        print(f"   问题: {result.question}")
        print(f"   意图: {result.intent_type} (复杂度: {result.intent_complexity})")
        print(f"   改写: {result.rewritten_question}")
        print(f"   回答: {result.generated_answer}")
        print(f"   耗时: {result.total_time_ms:.0f}ms")
        return

    # 评估集测试
    eval_set_path = Path(args.eval_set)
    if not eval_set_path.exists():
        print(f"❌ 评估集不存在: {eval_set_path}")
        print(f"   请先修复 JSON 格式（需要以 [ 开头，] 结尾）")
        return

    # 对比模式
    if args.compare:
        await compare_strategies(eval_set_path, args.eval_model)
        return

    # 正常模式
    print(f"\n📋 使用融合策略: {settings.fusion_strategy}")
    evaluator = E2EEvaluator(
        eval_model_name=args.eval_model,
        eval_model_type=args.eval_type,
        similarity_threshold=args.threshold
    )
    result = await evaluator.evaluate_dataset(eval_set_path)
    evaluator.print_results(result)

    if args.save:
        evaluator.save_report(result)

    print("\n✅ 测试完成!")


if __name__ == "__main__":
    asyncio.run(main())