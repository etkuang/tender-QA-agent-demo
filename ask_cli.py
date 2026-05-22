#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
命令行交互式问答客户端（配置化版本）
使用方式：
1. 确保 main.py 服务已经启动（运行 python main_agent.py）
2. 在这个脚本所在的终端执行：python ask_cli.py
3. 输入问题后按回车，等待返回答案
4. 输入 'exit' 或 'quit' 退出
"""

import httpx
import json
import sys
import os
from typing import Optional
from pathlib import Path

# 尝试加载配置文件（如果存在）
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from config import settings

    # 从配置读取API地址
    API_BASE_URL = f"http://localhost:{settings.port}"
except ImportError:
    # 降级：从环境变量读取
    API_HOST = os.getenv("API_HOST", "localhost")
    API_PORT = os.getenv("API_PORT", "8000")
    API_BASE_URL = f"http://{API_HOST}:{API_PORT}"

# 允许环境变量完全覆盖
API_BASE_URL = os.getenv("API_BASE_URL", API_BASE_URL)

# CLI配置
DEFAULT_TIMEOUT = float(os.getenv("CLI_TIMEOUT", "30.0"))
SHOW_SOURCES = os.getenv("CLI_SHOW_SOURCES", "true").lower() == "true"
SHOW_TIMING = os.getenv("CLI_SHOW_TIMING", "true").lower() == "true"
MAX_SOURCES_DISPLAY = int(os.getenv("CLI_MAX_SOURCES", "3"))


def ask_question(question: str, session_id: Optional[str] = None, timeout: float = DEFAULT_TIMEOUT) -> dict:
    """向 API 发送问题，返回完整响应"""
    payload = {"question": question}
    if session_id:
        payload["session_id"] = session_id

    try:
        response = httpx.post(
            f"{API_BASE_URL}/api/v1/ask",
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        return response.json()
    except httpx.ConnectError:
        return {"error": f"无法连接到服务，请确认 {API_BASE_URL} 上的服务是否已启动"}
    except httpx.TimeoutException:
        return {"error": f"请求超时（{timeout}秒），可能是服务处理时间过长或模型调用无响应"}
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP 错误 {e.response.status_code}: {e.response.text}"}
    except Exception as e:
        return {"error": f"未知错误: {str(e)}"}


def check_health() -> bool:
    """检查服务健康状态"""
    try:
        response = httpx.get(f"{API_BASE_URL}/api/v1/health", timeout=5.0)
        if response.status_code == 200:
            data = response.json()
            stats = data.get("stats", {})
            print(f"✅ 服务健康检查通过")
            print(f"   Regulations库: {stats.get('regulations', 0)} 条")

            return True
        else:
            print(f"⚠️ 服务状态异常，状态码: {response.status_code}")
            return False
    except Exception as e:
        print(f"⚠️ 无法获取服务健康状态: {e}")
        return False


def print_config_info():
    """打印当前配置信息"""
    print(f"📡 API地址: {API_BASE_URL}")
    print(f"⏱️  超时设置: {DEFAULT_TIMEOUT}秒")
    print(f"📚 显示来源: {'是' if SHOW_SOURCES else '否'}")
    print(f"⏱️  显示耗时: {'是' if SHOW_TIMING else '否'}")
    print(f"📖 最多显示来源数: {MAX_SOURCES_DISPLAY}")


def display_answer(result: dict):
    """显示答案和相关信息"""
    if "error" in result:
        print(f"❌ 错误: {result['error']}")
        if "detail" in result:
            print(f"   详情: {result['detail']}")
        return

    # 获取响应内容
    answer = result.get("answer", "没有返回答案")
    sources = result.get("sources", [])
    processing_time = result.get("processing_time", 0)
    session_id = result.get("session_id", "")
    route = result.get("route", "unknown")

    # 显示路由信息（调试用）
    route_icon = {
        "greeting": "👋",
        "rejected": "🚫",
        "rag": "⚡",
        "agent": "🧠"
    }.get(route, "❓")

    print(f"\n{route_icon} 答案: {answer}")

    # 显示来源
    if SHOW_SOURCES and sources:
        print(f"\n📚 参考来源 (共{len(sources)}条):")
        for i, src in enumerate(sources[:MAX_SOURCES_DISPLAY], 1):
            title = src.get("title", "无标题")
            content_preview = src.get("content_preview", "")[:100]
            print(f"  [{i}] {title}")
            if content_preview:
                print(f"      {content_preview}...")
        if len(sources) > MAX_SOURCES_DISPLAY:
            print(f"  ... 还有 {len(sources) - MAX_SOURCES_DISPLAY} 条未显示")

    # 显示耗时
    if SHOW_TIMING:
        print(f"\n⏱️  处理耗时: {processing_time:.2f} 秒")

    # 显示路由（调试用，可选）
    if os.getenv("CLI_DEBUG", "false").lower() == "true":
        print(f"🔍 路由模式: {route}")
        print(f"🆔 会话ID: {session_id}")


def print_welcome():
    """打印欢迎信息"""
    print("=" * 60)
    print("招投标智能问答系统 - 命令行客户端 (配置化版本)")
    print("=" * 60)
    print_config_info()
    print("=" * 60)
    print("📝 提问: 直接输入问题")
    print("-" * 60)
    print("🛠️ 命令:")
    print("   exit/quit/q  - 退出程序")
    print("   clear  - 清屏")
    print("   new        - 开启新会话（清空历史）")
    print("   info       - 查看当前会话信息")
    print("   stats      - 查看服务端会话统计")
    print("   cleanup    - 手动清理过期会话")
    print("   health     - 检查服务状态")
    print("   config     - 显示配置信息")
    print("=" * 60)


def main():
    print_welcome()

    # 可选：先检查服务健康状态
    if os.getenv("CLI_CHECK_HEALTH", "true").lower() == "true":
        check_health()
        print()

    session_id = None

    while True:
        try:
            user_input = input("\n💬 请输入问题: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 退出")
            break

        if user_input.lower() in ("exit", "quit", "q"):
            print("👋 退出")
            break

        if not user_input:
            print("⚠️ 问题不能为空")
            continue

        # 特殊命令
        if user_input.lower() == "health":
            check_health()
            continue

        if user_input.lower() == "config":
            print_config_info()
            continue
        # 查看当前会话信息
        if user_input.lower() == "info":
            if session_id:
                try:
                    response = httpx.get(f"{API_BASE_URL}/api/v1/session/{session_id}/info", timeout=5.0)
                    if response.status_code == 200:
                        info = response.json()
                        print(f"\n📜 当前会话:")
                        print(f"   消息数: {info.get('message_count', 0)}")
                        print(f"   空闲: {info.get('idle_hours', 0)} 小时")
                    else:
                        print("⚠️ 获取会话信息失败")
                except Exception as e:
                    print(f"⚠️ 获取失败: {e}")
            else:
                print("📜 当前无会话（新会话将在首次提问时创建）")
            continue

        if user_input.lower() == "clear": #clear 只是清屏（类似终端的 cls 或 clear 命令），跟会话管理完全无关
            os.system('cls' if os.name == 'nt' else 'clear')
            print_welcome()
            continue
        if user_input.lower() == "new": #开启新会话（丢弃当前历史）	客户端 + 服务端
            session_id = None
            print("🆕 已开启新会话，历史已清空")
            continue

        if user_input.lower() == "stats": #查看全局会话统计	服务端
            try:
                response = httpx.get(f"{API_BASE_URL}/api/v1/session/stats", timeout=5.0)
                if response.status_code == 200:
                    stats = response.json()
                    print(f"\n📊 会话统计:")
                    print(f"   活跃会话数: {stats.get('active_sessions', 0)}")
                    print(f"   每会话最大消息数: {stats.get('max_messages_per_session', 30)}")
                    print(f"   TTL: {stats.get('ttl_hours', 168)} 小时")
                    print(f"   清理间隔: {stats.get('cleanup_interval_minutes', 60)} 分钟")
                else:
                    print(f"⚠️ 获取统计失败: {response.status_code}")
            except Exception as e:
                print(f"⚠️ 获取统计失败: {e}")
            continue

        if user_input.lower() == "cleanup": #手动清理过期会话（服务端）
            try:
                response = httpx.post(f"{API_BASE_URL}/api/v1/session/cleanup", timeout=10.0)
                if response.status_code == 200:
                    result = response.json()
                    print(f"🗑️ 已清理 {result.get('cleaned', 0)} 个过期会话")
                    print(f"📊 当前活跃会话: {result.get('active', 0)}")
                else:
                    print(f"⚠️ 清理失败: {response.status_code}")
            except Exception as e:
                print(f"⚠️ 清理失败: {e}")
            continue
        print("⏳ 正在思考...")
        result = ask_question(user_input, session_id)

        display_answer(result)

        # 更新session_id以保持对话上下文
        if "session_id" in result and result["session_id"]:
            session_id = result["session_id"]

        print("-" * 60)


if __name__ == "__main__":
    main()