#!/usr/bin/env python
"""初始化招标库 - 适配新的Excel格式"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.config import settings
from app.storage.chroma_store import ChromaStore
from collections import defaultdict


def build_text_for_bid(record: dict) -> str:
    """
    构建检索文本 - 适配招标采购标的物信息提取训练数据.xlsx

    字段说明：
    - 项目名称: 核心检索字段
    - 标题: 辅助检索
    - 采购人/代理机构: 用户常问
    - 中标人/中标金额: 核心信息
    - 类别/省份/市区: 筛选维度
    """
    parts = [
        str(record.get("项目名称", "")),
        str(record.get("标题", "")),
        str(record.get("采购人", "")),
        str(record.get("代理机构", "")),
        str(record.get("中标人", "")),
        str(record.get("中标金额", "")),
        str(record.get("类别", "")),
        str(record.get("省份", "")),
        str(record.get("市区", "")),
        str(record.get("项目编号", "")),
        str(record.get("预算", "")),
    ]
    # 过滤空值和nan
    cleaned = []
    for p in parts:
        if p and str(p) != "nan" and str(p) != "":
            cleaned.append(str(p))
    return " ".join(cleaned)


def build_metadata_for_bid(record: dict) -> dict:
    """
    构建元数据 - 用于返回给用户展示
    """
    # 处理中标金额，转换为浮点数
    winner_amount = record.get("中标金额")
    if winner_amount and str(winner_amount) != "nan":
        try:
            winner_amount = float(winner_amount)
        except:
            winner_amount = str(winner_amount)

    budget = record.get("预算")
    if budget and str(budget) != "nan":
        try:
            budget = float(budget)
        except:
            budget = str(budget)

    return {
        "title": str(record.get("标题", "")),
        "project_name": str(record.get("项目名称", "")),
        "procuring_entity": str(record.get("采购人", "")),
        "procuring_agent": str(record.get("代理机构", "")),
        "winner": str(record.get("中标人", "")),
        "winner_amount": winner_amount,
        "category": str(record.get("类别", "")),
        "province": str(record.get("省份", "")),
        "city": str(record.get("市区", "")),
        "county": str(record.get("县城", "")),
        "project_no": str(record.get("项目编号", "")),
        "budget": budget,
        "publish_time": str(record.get("发布时间", "")),
        "winner_time": str(record.get("中标时间", "")),
        "source_type": "bid",
    }


def main():
    print("=" * 60)
    print("初始化招标库")
    print("=" * 60)

    # 使用你提供的文件名
    excel_path = "./data/招标采购标的物信息提取训练数据.xlsx"

    # 如果没有这个文件，尝试查找
    if not Path(excel_path).exists():
        # 可能文件在根目录
        alt_path = "../招标采购标的物信息提取训练数据.xlsx"
        if Path(alt_path).exists():
            excel_path = alt_path
        else:
            print(f"错误: 找不到文件")
            print(f"  尝试路径1: {excel_path}")
            print(f"  尝试路径2: {alt_path}")
            print("\n请将Excel文件放到 data/ 目录下，或修改 excel_path 变量")
            return

    print(f"\n读取文件: {excel_path}")

    import pandas as pd
    df = pd.read_excel(excel_path, sheet_name="Sheet2")  # 你的数据在Sheet2

    print(f"共读取 {len(df)} 行数据")
    print(f"字段列表: {df.columns.tolist()}")

    # 转换为字典列表
    data = df.to_dict('records')

    # 过滤有效数据（至少要有项目名称或中标人）
    valid_data = []
    for record in data:
        project_name = record.get("项目名称", "")
        winner = record.get("中标人", "")
        if project_name and str(project_name) != "nan":
            valid_data.append(record)
        elif winner and str(winner) != "nan":
            valid_data.append(record)

    print(f"有效数据: {len(valid_data)} 条")

    if not valid_data:
        print("错误: 没有有效数据")
        return

    # 构建文本和元数据
    texts = []
    metadatas = []
    ids = []

    # 用于记录每个项目编号出现的次数，用于生成唯一ID
    id_counter = defaultdict(int)

    for i, record in enumerate(valid_data):
        text = build_text_for_bid(record)
        if len(text) < 10:  # 跳过文本太短的
            continue

        metadata = build_metadata_for_bid(record)

        # 生成唯一ID - 使用项目编号 + 序号避免重复
        project_no = record.get("项目编号", "")
        if project_no and str(project_no) != "nan":
            # 同一项目可能有多条（不同中标人），加序号
            id_counter[str(project_no)] += 1
            doc_id = f"bid_{project_no}_{id_counter[str(project_no)]}"
        else:
            doc_id = f"bid_{i}_{hash(text) % 100000}"

        texts.append(text)
        metadatas.append(metadata)
        ids.append(doc_id)

    print(f"待入库: {len(texts)} 条")

    # 初始化Chroma并入库
    client = ChromaStore()

    # 删除旧库重建
    print("\n删除旧的 bids 库...")
    try:
        client.delete_collection("bids")
    except:
        print("  库不存在，跳过删除")

    print("开始导入...")

    # 分批导入，避免一次性导入过多
    batch_size = 500
    total = len(texts)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        print(f"  导入 {start + 1} - {end} / {total}")
        client.add_documents(
            "bids",
            texts[start:end],
            metadatas[start:end],
            ids[start:end]
        )

    print("\n" + "=" * 60)
    print(f"初始化完成!")
    print(f"  Bids库: {client.get_count('bids')} 条")
    print("=" * 60)

    # 显示样例
    print("\n样例数据（前3条）:")
    for i in range(min(3, len(texts))):
        print(f"  [{i + 1}] {metadatas[i].get('project_name', '')[:50]}...")
        print(f"      中标人: {metadatas[i].get('winner', '未知')}")
        print(f"      金额: {metadatas[i].get('winner_amount', '未知')}")


if __name__ == "__main__":
    main()