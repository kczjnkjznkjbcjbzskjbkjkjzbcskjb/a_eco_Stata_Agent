"""
main.py（Diamond稳定版）

特点：
- 真正可调试
- 支持 LangGraph Agent
- 支持 ToolMessage 解析
- 支持 MCP 调试
- 支持错误定位
- 支持完整执行日志
- 支持最终状态分析
"""

import asyncio
import logging
import sys
import os
import json

# =========================================================
# 确保项目根目录在 Python 路径中
# =========================================================

sys.path.insert(
    0,
    os.path.dirname(os.path.abspath(__file__))
)

# =========================================================
# imports
# =========================================================

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
)

from tools.MCP_stata import StataMCPManager
from tools.stata_tools import create_tool_node
from llm.client import DualLLMClient
from agent.graph import StataAgent


logger = logging.getLogger(__name__)


# =========================================================
# UI helpers
# =========================================================

def print_separator(title: str = ""):
    """打印分隔线"""

    if title:
        print(f"\n{'=' * 70}")
        print(f"  {title}")
        print(f"{'=' * 70}")
    else:
        print(f"{'─' * 70}")


def shorten(text: str, limit: int = 1200) -> str:
    """截断长文本"""

    text = str(text)

    if len(text) <= limit:
        return text

    return (
        text[:limit]
        + f"\n...（已截断，原始长度 {len(text)} 字符）"
    )


def parse_tool_message(content):
    """
    尝试解析 MCP 返回内容
    """

    try:

        if isinstance(content, list):

            if len(content) > 0:

                item = content[0]

                if isinstance(item, dict):

                    if "text" in item:
                        return json.loads(item["text"])

        if isinstance(content, str):

            data = json.loads(content)

            if isinstance(data, list):

                item = data[0]

                if isinstance(item, dict) and "text" in item:
                    return json.loads(item["text"])

            return data

    except Exception:
        pass

    return None


def print_message(msg, index: int):
    """
    格式化打印消息
    """

    msg_type = type(msg).__name__

    icons = {
        "HumanMessage": "👤",
        "AIMessage": "🤖",
        "ToolMessage": "🔧",
    }

    icon = icons.get(msg_type, "📌")

    print(f"\n{icon} [{index}] {msg_type}")
    print("─" * 70)

    # =====================================================
    # ToolMessage 特殊解析
    # =====================================================

    if isinstance(msg, ToolMessage):

        content_str = str(msg.content)

        # 文件工具结果（纯文本）
        if "文件已创建" in content_str:
            print("📄 文件创建成功")
            print(shorten(content_str))
            return

        if "文件已修改" in content_str:
            print("✏️  文件修改成功")
            print(shorten(content_str))
            return

        if "读取文件成功" in content_str:
            print("📖 文件读取成功")
            print(shorten(content_str))
            return

        # MCP / Stata 工具结果（JSON）
        parsed = parse_tool_message(msg.content)

        if parsed:

            ok = parsed.get("ok", False)

            print(f"工具执行状态: {'✅ SUCCESS' if ok else '❌ FAILED'}")

            if not ok:

                rc = parsed.get("rc")

                if rc is not None:
                    print(f"Stata 返回码: {rc}")

                log = parsed.get("log", {})

                if isinstance(log, dict):

                    tail = log.get("tail", "")

                    if tail:
                        print("\n错误日志:")
                        print(tail)

                stderr = parsed.get("stderr", "")

                if stderr:
                    print("\nstderr:")
                    print(stderr)

            else:

                results = parsed.get("results", {})

                if results:
                    print("\n结果摘要:")
                    print(shorten(json.dumps(results, indent=2, ensure_ascii=False)))

            return

    # =====================================================
    # 普通消息
    # =====================================================

    content = shorten(msg.content)

    print(content)


# =========================================================
# final status analysis
# =========================================================

def analyze_final_status(result: dict):
    """
    判断 Agent 是否真正成功

    遍历所有 ToolMessage，只要有一个成功就算成功。
    这避免了多轮交互（read_file → stata_run）时只检查最后一轮的问题。
    """
    messages = result.get("messages", [])

    # 收集所有 ToolMessage
    tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
    if not tool_msgs:
        return False, "没有 ToolMessage"

    # 先检查是否有文件操作成功的消息（优先级最高）
    for msg in tool_msgs:
        content_str = str(msg.content)
        if any(kw in content_str for kw in ["文件已创建", "文件已修改"]):
            return True, "文件操作成功"

    # 检查是否有 read_file 成功 + 后续 stata_run 成功的组合
    # 从后往前找最后一个有实际结果的 ToolMessage
    for msg in reversed(tool_msgs):
        content_str = str(msg.content)

        # 跳过纯文件读取成功消息（它们只是中间步骤）
        if "读取文件成功" in content_str:
            continue

        # 尝试解析 JSON
        parsed = parse_tool_message(msg.content)
        if parsed:
            ok = parsed.get("ok", False)
            if ok:
                return True, "执行成功"
            rc = parsed.get("rc")
            log = parsed.get("log", {})
            tail = ""
            if isinstance(log, dict):
                tail = log.get("tail", "")
            return False, f"rc={rc} {tail}"

        # 无法解析，检查是否是错误信息
        if "error" in content_str.lower() or "r(" in content_str:
            return False, content_str[:200]

    # 如果能走到这里，只有 read_file 成功的消息 → 成功
    for msg in tool_msgs:
        if "读取文件成功" in str(msg.content):
            return True, "文件读取成功"

    return False, "无法解析工具结果"


# =========================================================
# interactive query
# =========================================================

async def interactive_loop(agent):

    while True:

        print_separator("输入查询")

        query = input("👤 请输入需求（输入 exit 退出）:\n> ").strip()

        if query.lower() in ["exit", "quit"]:
            break

        if not query:
            continue

        print_separator("开始执行")

        try:

            result = await agent.invoke(query)

            # =================================================
            # 执行过程
            # =================================================

            print_separator("执行过程")

            for i, msg in enumerate(result["messages"]):

                print_message(msg, i)

            # =================================================
            # 执行摘要
            # =================================================

            print_separator("执行摘要")

            print(f"总消息数: {len(result['messages'])}")

            print(f"重试次数: {result.get('retry_count', 0)}")

            print("\n最终代码:")
            print("─" * 70)

            print(result.get("current_code", "N/A"))

            print("─" * 70)

            # =================================================
            # 最终状态分析
            # =================================================

            success, reason = analyze_final_status(result)

            if success:

                print("\n状态: ✅ 执行成功")

            else:

                print("\n状态: ❌ 执行失败")

                print(f"原因: {reason}")

        except Exception as e:

            print(f"\n❌ Agent 执行异常: {e}")

            import traceback

            traceback.print_exc()


# =========================================================
# main
# =========================================================

async def main():

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-5s | %(message)s',
        datefmt='%H:%M:%S',
    )

    print_separator("Stata Code Agent v2.0")

    print("基于 LangGraph + MCP + DeepSeek 的工业级 Agent")

    print_separator()

    try:

        # =====================================================
        # step 1
        # =====================================================

        print("\n[1/3] 初始化 LLM 客户端...")

        llm = DualLLMClient()

        print("      ✅ DeepSeek Client 已就绪")

        # =====================================================
        # step 2
        # =====================================================

        print("\n[2/3] 连接 Stata MCP 服务...")

        async with StataMCPManager() as manager:

            raw_tools = await manager.load_tools()

            print(f"      ✅ 成功加载 {len(raw_tools)} 个 MCP 工具")

            print("\nMCP 工具列表:")

            for tool in raw_tools:

                print(f"  - {tool.name}")

            tool_node = create_tool_node(raw_tools)

            # 展示完整工具列表（MCP + 文件工具）
            all_tool_names = list(tool_node.tools_by_name.keys())
            file_tool_names = [n for n in all_tool_names if n not in {"stata_run", "get_log"}]
            if file_tool_names:
                print(f"\n📁 文件操作工具 ({len(file_tool_names)} 个):")
                for name in file_tool_names:
                    print(f"  - {name}")
                print(f"\n  共 {len(all_tool_names)} 个工具可用")

            # =================================================
            # step 3
            # =================================================

            print("\n[3/3] 构建 Agent 工作流...")

            agent = StataAgent(tool_node, llm)

            print("      ✅ Agent 已构建")

            print("\n工作流:")
            print("generate → prepare_tools → execute_tools → review ↻")
            print("              ↑ (文件操作成功后自动回链) ========┘")

            # =================================================
            # interactive loop
            # =================================================

            await interactive_loop(agent)

    except Exception as e:

        print(f"\n❌ 系统启动失败: {e}")

        import traceback

        traceback.print_exc()

    print_separator("运行结束")


# =========================================================
# entry
# =========================================================

if __name__ == "__main__":

    asyncio.run(main())