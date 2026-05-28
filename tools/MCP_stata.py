"""
Stata MCP 连接管理器
负责：启动服务、建立连接、加载工具、资源清理
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

logger = logging.getLogger(__name__)

# ==================== 默认配置 ====================
DEFAULT_COMMAND = "stata-code-mcp"
DEFAULT_ARGS: list = []
CONNECTION_TIMEOUT = 30  # 连接超时（秒）
MAX_RETRIES = 3          # 连接失败最大重试次数


class StataMCPManager:
    """
    Stata MCP 连接管理器
    封装了服务端启动、客户端连接、工具加载和资源清理的完整生命周期
    """

    def __init__(self, work_dir: str | None = None, command: str = DEFAULT_COMMAND):
        """
        参数:
            work_dir: Stata 工作目录，决定 do 文件和日志的存放位置
            command:  启动 MCP 服务端的命令，默认 stata-code-mcp
        """
        self.work_dir = work_dir
        self.command = command
        self._server_params = self._build_params()
        self._tools: list | None = None
        # 保存读写流和会话的引用，用于手动管理生命周期
        self._read = None
        self._write = None
        self._session = None
        self._context_stack = None

    def _build_params(self) -> StdioServerParameters:
        """构建服务启动参数"""
        env = None
        if self.work_dir:
            env = {"STATA_MCP_CWD": self.work_dir}
        return StdioServerParameters(
            command=self.command,
            args=DEFAULT_ARGS,
            env=env
        )

    async def connect(self) -> None:
        """
        建立与 MCP 服务端的连接
        支持失败重试，重试次数由 MAX_RETRIES 控制
        """
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(f"正在连接 Stata MCP 服务... (第 {attempt}/{MAX_RETRIES} 次)")
                # 设置连接超时
                self._context_stack = stdio_client(self._server_params)
                self._read, self._write = await asyncio.wait_for(
                    self._context_stack.__aenter__(),
                    timeout=CONNECTION_TIMEOUT
                )
                self._session = ClientSession(self._read, self._write)
                await asyncio.wait_for(
                    self._session.__aenter__(),
                    timeout=CONNECTION_TIMEOUT
                )
                await asyncio.wait_for(
                    self._session.initialize(),
                    timeout=CONNECTION_TIMEOUT
                )
                logger.info("Stata MCP 服务连接成功")
                return
            except asyncio.TimeoutError:
                last_error = TimeoutError(f"连接超时 ({CONNECTION_TIMEOUT}秒)")
                logger.warning(f"连接超时，第 {attempt} 次尝试失败")
            except Exception as e:
                last_error = e
                logger.warning(f"连接失败: {e}")
            # 失败后等待 2 秒再重试
            if attempt < MAX_RETRIES:
                await asyncio.sleep(2)
        raise ConnectionError(f"无法连接 Stata MCP 服务，已重试 {MAX_RETRIES} 次") from last_error

    async def load_tools(self) -> list:
        """加载所有 MCP 工具并返回 LangChain 兼容的工具列表"""
        if self._session is None:
            raise RuntimeError("请先调用 connect() 建立连接")
        if self._tools is None:
            logger.info("正在加载 Stata MCP 工具...")
            self._tools = await load_mcp_tools(self._session)
            logger.info(f"成功加载 {len(self._tools)} 个工具")
        return self._tools

    async def disconnect(self) -> None:
        """断开连接并释放资源"""
        if self._session is not None:
            try:
                await self._session.__aexit__(None, None, None)
            except Exception:
                pass
            self._session = None
        if self._context_stack is not None:
            try:
                await self._context_stack.__aexit__(None, None, None)
            except Exception:
                pass
            self._context_stack = None
        self._read = None
        self._write = None
        self._tools = None
        logger.info("Stata MCP 连接已断开")

    async def __aenter__(self):
        """上下文管理器入口：自动建立连接"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口：自动释放资源"""
        await self.disconnect()
        return False

    def get_tools(self) -> list:
        """同步获取已加载的工具列表（需先调用 connect + load_tools）"""
        if self._tools is None:
            raise RuntimeError("工具尚未加载，请先调用 connect() 和 load_tools()")
        return self._tools


# ==================== 便捷函数 ====================

async def load_stata_tools(work_dir: str | None = None) -> list:
    """
    一键加载 Stata 工具的便捷函数
    内部使用 StataMCPManager，返回工具列表
    注意：返回的工具需要在 manager 存活期间使用
    """
    manager = StataMCPManager(work_dir=work_dir)
    await manager.connect()
    tools = await manager.load_tools()
    # 将 manager 附加到工具列表上，防止被垃圾回收
    tools._manager = manager
    return tools


# ==================== 测试入口 ====================

async def main():
    """测试 StataMCPManager 的连接和工具加载"""
    print("=" * 50)
    print("测试 StataMCPManager")
    print("=" * 50)

    # 方式1：上下文管理器（推荐）
    print("\n[方式1] 使用 async with 上下文管理器:")
    async with StataMCPManager() as manager:
        tools = await manager.load_tools()
        print(f"加载了 {len(tools)} 个工具:")
        for t in tools[:5]:  # 只打印前5个
            desc = (t.description or "")[:60]
            print(f"  - {t.name}: {desc}...")
        if len(tools) > 5:
            print(f"  ... 以及其他 {len(tools) - 5} 个工具")

    # 方式2：手动管理生命周期
    print("\n[方式2] 手动管理连接:")
    manager = StataMCPManager()
    try:
        await manager.connect()
        tools = await manager.load_tools()
        print(f"手动模式下加载了 {len(tools)} 个工具")
    finally:
        await manager.disconnect()

    print("\n测试完成")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    asyncio.run(main())