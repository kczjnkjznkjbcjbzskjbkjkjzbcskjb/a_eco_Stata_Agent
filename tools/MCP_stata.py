"""
Stata MCP 连接管理器（Diamond 级稳定版）
负责：
- MCP 生命周期管理
- 工具加载
- 连接恢复
- 结构化输出兼容
"""

import asyncio
import logging
from contextlib import AsyncExitStack
from typing import Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

logger = logging.getLogger(__name__)


# ==================== 配置 ====================
DEFAULT_COMMAND = "stata-code-mcp"
DEFAULT_ARGS = []
CONNECTION_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_INTERVAL = 2


# ==================== Manager ====================
class StataMCPManager:
    """
    Diamond级 MCP 管理器
    特点：
    - 单一 AsyncExitStack 管理生命周期
    - 自动重连
    - 工具标准化
    """

    def __init__(self, work_dir: str | None = None, command: str = DEFAULT_COMMAND):
        self.work_dir = work_dir
        self.command = command

        self._server_params = self._build_params()

        self._stack = AsyncExitStack()
        self._session: Optional[ClientSession] = None
        self._tools: Optional[list] = None

    # ==================== params ====================
    def _build_params(self) -> StdioServerParameters:
        env = {"STATA_MCP_CWD": self.work_dir} if self.work_dir else None

        return StdioServerParameters(
            command=self.command,
            args=DEFAULT_ARGS,
            env=env,
        )

    # ==================== connect ====================
    async def connect(self) -> None:
        """
        带重试的 MCP 连接
        """

        last_error = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(f"[MCP] 连接中... {attempt}/{MAX_RETRIES}")

                read, write = await asyncio.wait_for(
                    self._stack.enter_async_context(
                        stdio_client(self._server_params)
                    ),
                    timeout=CONNECTION_TIMEOUT,
                )

                self._session = ClientSession(read, write)

                await asyncio.wait_for(
                    self._stack.enter_async_context(self._session),
                    timeout=CONNECTION_TIMEOUT,
                )

                await asyncio.wait_for(
                    self._session.initialize(),
                    timeout=CONNECTION_TIMEOUT,
                )

                logger.info("[MCP] 连接成功")
                return

            except Exception as e:
                last_error = e
                logger.warning(f"[MCP] 连接失败: {e}")

                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_INTERVAL)

        raise ConnectionError(
            f"MCP连接失败，已重试{MAX_RETRIES}次"
        ) from last_error

    # ==================== tools ====================
    async def load_tools(self) -> list:
        """
        加载 MCP tools（统一结构）
        """

        if self._session is None:
            raise RuntimeError("请先 connect()")

        if self._tools is None:
            logger.info("[MCP] 加载工具中...")

            raw_tools = await load_mcp_tools(self._session)

            # ==============================
            # 🔥 关键增强：工具包装层
            # ==============================
            self._tools = [
                self._wrap_tool(t) for t in raw_tools
            ]

            logger.info(f"[MCP] 加载完成: {len(self._tools)} tools")

        return self._tools

    # ==================== tool wrapper ====================
    def _wrap_tool(self, tool):
        """
        标准化 tool 输出（关键修复点）
        """

        async def safe_ainvoke(**kwargs):
            try:
                result = await tool.ainvoke(kwargs)

                return {
                    "ok": True,
                    "tool": tool.name,
                    "result": result,
                    "error": None,
                }

            except Exception as e:
                return {
                    "ok": False,
                    "tool": tool.name,
                    "result": None,
                    "error": str(e),
                }

        tool.ainvoke = safe_ainvoke
        return tool

    # ==================== disconnect ====================
    async def disconnect(self) -> None:
        try:
            await self._stack.aclose()
        except Exception:
            pass

        self._session = None
        self._tools = None

        logger.info("[MCP] 已断开连接")

    # ==================== context ====================
    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.disconnect()
        return False

    # ==================== tools getter ====================
    def get_tools(self) -> list:
        if self._tools is None:
            raise RuntimeError("tools 未加载")
        return self._tools


# ==================== convenience ====================
async def load_stata_tools(work_dir: str | None = None) -> list:
    """
    一键加载工具（安全版）
    """

    manager = StataMCPManager(work_dir=work_dir)

    await manager.connect()
    tools = await manager.load_tools()

    # ⚠️ 不再暴露 hidden manager
    return tools


# ==================== test ====================
async def main():
    async with StataMCPManager() as m:
        tools = await m.load_tools()
        print(f"tools: {len(tools)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())