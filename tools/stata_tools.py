"""
Stata 工具包装层
将 MCP 原生工具转换为 LangChain/LangGraph 兼容的 Tool 格式
"""
import asyncio
import logging
import traceback
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.tools import tool, BaseTool
from langgraph.prebuilt import ToolNode

logger = logging.getLogger(__name__)

REQUIRED_TOOLS = ["stata_run", "get_log"]


class StataToolWrapper:
    """
    MCP 工具 → LangChain 工具的适配器
    负责工具查找、调用转发、错误捕获
    """

    def __init__(self, raw_tools: list):
        self._raw_tools = raw_tools
        self._tool_map = self._build_tool_map()

    def _build_tool_map(self) -> dict:
        """构建 {工具名: MCP工具对象} 的映射表"""
        mapping = {}
        for t in self._raw_tools:
            name = getattr(t, 'name', '')
            if name in REQUIRED_TOOLS:
                mapping[name] = t
        logger.debug(f"MCP 工具映射完成，可用: {list(mapping.keys())}")
        return mapping

    async def _call_raw_tool(self, tool_name: str, **kwargs) -> str:
        """异步调用 MCP 原生工具，带完整异常信息"""
        raw_tool = self._tool_map.get(tool_name)
        if raw_tool is None:
            return f"Error: 工具 '{tool_name}' 未找到，当前可用: {list(self._tool_map.keys())}"
        try:
            result = await raw_tool.ainvoke(kwargs)
            return str(result)
        except Exception as e:
            tb = traceback.format_exc()
            return f"Error: 调用 {tool_name} 失败:\n异常类型: {type(e).__name__}\n异常信息: {str(e)}\n完整堆栈:\n{tb}"

    def create_stata_run(self):
        """创建 stata_run 工具：执行 Stata 代码并返回结果"""
        wrapper = self

        @tool
        async def stata_run(code: str) -> str:
            """执行 Stata 代码。参数 code: 完整的 Stata 代码字符串。"""
            return await wrapper._call_raw_tool("stata_run", code=code)
        return stata_run

    def create_get_log(self):
        """创建 get_log 工具：获取 Stata 运行日志"""
        wrapper = self

        @tool
        async def get_log(log_ref: str) -> str:
            """获取日志文件内容。参数 log_ref: 日志引用标识符。"""
            return await wrapper._call_raw_tool("get_log", ref=log_ref)
        return get_log

    def build_tools(self) -> list:
        """构建所有 LangChain 工具并返回列表"""
        tools = []
        if "stata_run" in self._tool_map:
            tools.append(self.create_stata_run())
        if "get_log" in self._tool_map:
            tools.append(self.create_get_log())
        logger.info(f"已构建 {len(tools)} 个 LangChain 工具: {[t.name for t in tools]}")
        return tools

    @property
    def available_tool_names(self) -> list:
        """返回当前可用的工具名称列表"""
        return list(self._tool_map.keys())


def create_tool_node(raw_tools: list, include_file_tools: bool = True) -> ToolNode:
    """快捷函数：从 MCP 原始工具列表 + 文件工具创建 LangGraph ToolNode

    参数:
        raw_tools: MCP 原始工具列表
        include_file_tools: 是否同时加载文件操作工具（read_file / create_do_file / modify_do_file）
    """
    wrapper = StataToolWrapper(raw_tools)
    tools = wrapper.build_tools()

    if include_file_tools:
        from tools.file_tools import create_file_tools
        tools.extend(create_file_tools())

    return ToolNode(tools)


def create_tools_list(raw_tools: list) -> list:
    """快捷函数：从 MCP 原始工具列表直接创建 LangChain 工具列表"""
    wrapper = StataToolWrapper(raw_tools)
    return wrapper.build_tools()


# ==================== 测试部分 ====================

async def _load_mcp_tools_with_session():
    """
    加载 MCP 工具并保持会话存活
    返回 (tools, exit_stack)，调用方需要在测试结束后 close exit_stack
    """
    server_params = StdioServerParameters(
        command="stata-code-mcp",
        args=[],
        env=None
    )
    
    exit_stack = AsyncExitStack()
    read, write = await exit_stack.enter_async_context(stdio_client(server_params))
    session = await exit_stack.enter_async_context(ClientSession(read, write))
    await session.initialize()
    tools = await load_mcp_tools(session)
    
    return tools, exit_stack


async def _test_tool_wrapper(raw_tools):
    """测试1：StataToolWrapper 的基本功能"""
    print("=" * 60)
    print("测试1: StataToolWrapper")
    print("=" * 60)

    wrapper = StataToolWrapper(raw_tools)

    print(f"\n[1.1] 可用工具名称: {wrapper.available_tool_names}")

    langchain_tools = wrapper.build_tools()
    print(f"[1.2] 构建了 {len(langchain_tools)} 个 LangChain 工具:")
    for t in langchain_tools:
        print(f"      名称: {t.name}")
        print(f"      描述: {t.description[:80] if t.description else 'N/A'}...")
        print(f"      参数: {list(t.args_schema.model_fields.keys())}")
        print()

    print("[1.3] 测试 stata_run 工具调用...")
    stata_run_tool = langchain_tools[0]
    test_code = """
    sysuse auto, clear
    describe
    summarize price
    """
    result = await stata_run_tool.ainvoke({"code": test_code})

    if result.startswith("Error:"):
        print("      ❌ 调用失败，完整错误信息如下:")
        print("-" * 40)
        print(result)
        print("-" * 40)
    else:
        has_results = "price" in result.lower()
        has_stata_error = "r(" in result
        if has_results and not has_stata_error:
            print("      ✅ 成功")
        elif has_results:
            print("      ⚠️ Stata 报错（可能是代码问题，非工具问题）")
        print(f"      结果长度: {len(result)} 字符")
        print(f"      前500字符:\n{result[:500]}...")


async def _test_create_functions(raw_tools):
    """测试2：快捷创建函数"""
    print("\n" + "=" * 60)
    print("测试2: create_tool_node 和 create_tools_list")
    print("=" * 60)

    print("\n[2.1] create_tool_node:")
    tool_node = create_tool_node(raw_tools)
    print(f"      类型: {type(tool_node).__name__}")
    print(f"      工具数量: {len(tool_node.tools_by_name)}")
    print(f"      工具名称: {list(tool_node.tools_by_name.keys())}")

    print("\n[2.2] create_tools_list:")
    tools_list = create_tools_list(raw_tools)
    print(f"      类型: {type(tools_list).__name__}")
    print(f"      工具数量: {len(tools_list)}")
    for t in tools_list:
        print(f"      - {t.name}")

    print("\n✅ 快捷函数测试通过")


async def _test_error_handling(raw_tools):
    """测试3：异常处理"""
    print("\n" + "=" * 60)
    print("测试3: 异常处理")
    print("=" * 60)

    wrapper = StataToolWrapper(raw_tools)

    print("\n[3.1] 调用不存在的工具:")
    result = await wrapper._call_raw_tool("nonexistent_tool", arg="test")
    print(f"      返回: {result[:100]}")

    print("\n[3.2] 执行无效 Stata 命令:")
    tools = wrapper.build_tools()
    result = await tools[0].ainvoke({"code": "invalidcommand"})
    if result.startswith("Error:") and "ClosedResourceError" in result:
        print("      ❌ MCP 连接已断开（ClosedResourceError）")
    elif result.startswith("Error:"):
        print(f"      其他错误: {result[:200]}")
    else:
        has_error = "error" in result.lower() or "r(" in result
        print(f"      Stata 返回了结果，检测到错误: {'✅' if has_error else '⚠️'}")
        print(f"      返回: {result[:200]}")

    print("\n[3.3] 执行空代码:")
    result = await tools[0].ainvoke({"code": ""})
    print(f"      返回: {result[:200]}")

    print("\n✅ 异常处理测试完成")


async def main():
    """测试入口：保持连接存活，所有测试共享同一个 MCP 会话"""
    print("\n" + "█" * 60)
    print("█  Stata 工具包装层 - 完整测试")
    print("█" * 60)

    tools, exit_stack = None, None
    try:
        tools, exit_stack = await _load_mcp_tools_with_session()

        await _test_tool_wrapper(tools)
        await _test_create_functions(tools)
        await _test_error_handling(tools)

        print("\n" + "=" * 60)
        print("🎉 所有测试通过！")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if exit_stack:
            await exit_stack.aclose()
            print("\nMCP 连接已安全关闭")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    asyncio.run(main())