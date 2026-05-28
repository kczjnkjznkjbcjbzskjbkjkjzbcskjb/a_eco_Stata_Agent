"""
文件操作工具
提供读取文件、创建 .do 文件、修改 .do 文件的功能

所有工具均为 LangChain @tool，可直接加入 ToolNode 与 MCP 工具混用
"""
import os
import logging
from pathlib import Path
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
async def read_file(path: str) -> str:
    """读取任意文件的内容并返回。
    参数 path: 文件路径（支持相对路径和绝对路径）。
    """
    file_path = Path(path)
    if not file_path.exists():
        return f"Error: 文件不存在: {path}"
    if not file_path.is_file():
        return f"Error: 路径不是文件: {path}"

    try:
        # 尝试多种编码
        for encoding in ["utf-8-sig", "utf-8", "gbk", "gb2312", "latin-1"]:
            try:
                content = file_path.read_text(encoding=encoding)
                logger.info(f"读取文件成功: {path} ({len(content)} 字符, 编码: {encoding})")
                return content
            except UnicodeDecodeError:
                continue
        return f"Error: 无法解码文件 {path}，已尝试 utf-8 / gbk / gb2312 / latin-1"
    except Exception as e:
        return f"Error: 读取文件失败: {str(e)}"


@tool
async def create_do_file(path: str, content: str) -> str:
    """创建新的 Stata .do 文件并写入内容。如果文件已存在则报错。
    参数 path: 文件路径（建议以 .do 结尾）。
    参数 content: 要写入的 Stata 代码内容。
    """
    file_path = Path(path)

    # 安全检查：限制在项目目录内
    try:
        file_path = file_path.resolve()
    except Exception:
        pass

    if file_path.exists():
        return f"Error: 文件已存在: {file_path}。如需修改请使用 modify_do_file。"

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8-sig")
        logger.info(f"创建文件成功: {file_path} ({len(content)} 字符)")
        return f"文件已创建: {file_path}\n--- 内容 ---\n{content}"
    except Exception as e:
        return f"Error: 创建文件失败: {str(e)}"


@tool
async def modify_do_file(path: str, content: str) -> str:
    """修改（覆盖）已存在的 .do 文件。
    参数 path: 文件路径。
    参数 content: 新的完整文件内容。
    """
    file_path = Path(path)

    try:
        file_path = file_path.resolve()
    except Exception:
        pass

    if not file_path.exists():
        return f"Error: 文件不存在: {file_path}。如需创建新文件请使用 create_do_file。"

    try:
        file_path.write_text(content, encoding="utf-8-sig")
        logger.info(f"修改文件成功: {file_path} ({len(content)} 字符)")
        return f"文件已修改: {file_path}\n--- 新内容 ---\n{content}"
    except Exception as e:
        return f"Error: 修改文件失败: {str(e)}"


def create_file_tools() -> list:
    """创建所有文件操作工具的列表"""
    return [read_file, create_do_file, modify_do_file]


# ==================== 测试 ====================
if __name__ == "__main__":
    import asyncio

    async def test():
        test_dir = Path("./test_file_tools")
        test_dir.mkdir(exist_ok=True)
        test_file = test_dir / "test.do"

        # 清理
        if test_file.exists():
            test_file.unlink()

        # --- 测试 read_file 文件不存在 ---
        result = await read_file.ainvoke({"path": "nonexistent.xyz"})
        assert "不存在" in result
        print("✅ read_file 文件不存在 → 正确报错")

        # --- 测试 create_do_file ---
        content = "sysuse auto, clear\nreg price mpg"
        result = await create_do_file.ainvoke({"path": str(test_file), "content": content})
        assert "已创建" in result
        assert test_file.exists()
        print("✅ create_do_file → 成功创建")

        # --- 测试 create_do_file 文件已存在 ---
        result = await create_do_file.ainvoke({"path": str(test_file), "content": content})
        assert "已存在" in result
        print("✅ create_do_file 文件已存在 → 正确报错")

        # --- 测试 read_file ---
        result = await read_file.ainvoke({"path": str(test_file)})
        assert "sysuse auto" in result
        assert "reg price mpg" in result
        print("✅ read_file → 成功读取")

        # --- 测试 modify_do_file ---
        new_content = "sysuse auto, clear\nreg price mpg weight"
        result = await modify_do_file.ainvoke({"path": str(test_file), "content": new_content})
        assert "已修改" in result
        stored = test_file.read_text(encoding="utf-8-sig")
        assert "weight" in stored
        print("✅ modify_do_file → 成功修改")

        # --- 测试 modify_do_file 文件不存在 ---
        result = await modify_do_file.ainvoke({"path": "no_such_file.do", "content": "x"})
        assert "不存在" in result
        print("✅ modify_do_file 文件不存在 → 正确报错")

        # 清理
        test_file.unlink()
        test_dir.rmdir()
        print("\n🎉 file_tools 全部测试通过")

    asyncio.run(test())
