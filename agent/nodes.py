"""
图节点函数
负责：生成代码/决定动作 → 封装工具调用 → 执行工具 → 反思修正
"""
import re
from langchain_core.messages import AIMessage, ToolMessage
from agent.state import StataAgentState
from agent.prompts import CODER_SYSTEM_PROMPT, REVIEWER_SYSTEM_PROMPT
from utils.code_parser import extract_code


# =========================================================
# Action 指令解析
# =========================================================

_ACTION_RE = re.compile(r'\[ACTION:(\w+)\]\s*\n(.*)', re.DOTALL)
_KV_RE = re.compile(r'^(\w+):\s*(.+)$', re.MULTILINE)

# 需要在 generate → prepare 之间透传的文件工具参数名
_FILE_TOOL_PARAMS = {
    "read_file": ["path"],
    "create_do_file": ["path", "content"],
    "modify_do_file": ["path", "content"],
}


def parse_action_directive(text: str) -> dict | None:
    """解析 LLM 输出中的 [ACTION:xxx] 指令，返回 {"tool": str, "args": dict} 或 None"""
    m = _ACTION_RE.search(text)
    if not m:
        return None

    tool_name = m.group(1).strip()
    body = m.group(2).strip()

    if tool_name not in _FILE_TOOL_PARAMS:
        return None

    args = {}
    for kv in _KV_RE.finditer(body):
        key = kv.group(1).strip()
        value = kv.group(2).strip()
        if key in _FILE_TOOL_PARAMS[tool_name]:
            args[key] = value

    # 如果 body 中包含代码块，提取为 content
    if "content" in _FILE_TOOL_PARAMS[tool_name] and "content" not in args:
        code = extract_code(body)
        if code:
            args["content"] = code

    return {"tool": tool_name, "args": args}


# =========================================================
# 工具调用辅助
# =========================================================

async def _invoke_tool(tool_node, tool_name: str, **kwargs) -> str:
    """异步调用单个工具"""
    tools = tool_node.tools_by_name
    tool = tools.get(tool_name)
    if tool is None:
        return f"Error: 工具 {tool_name} 未找到"
    try:
        result = await tool.ainvoke(kwargs)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


# =========================================================
# 节点函数
# =========================================================

async def generate_code(state: StataAgentState, llm_client) -> dict:
    """节点1：根据用户需求生成 Stata 代码或决定文件操作"""
    user_query = state["messages"][-1].content
    response = await llm_client.call_coder([
        {"role": "system", "content": CODER_SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ])

    code = extract_code(response)

    # 解析 action 指令
    action = parse_action_directive(response)
    if action is None:
        # 无指令 → 默认执行 Stata 代码
        action = {"tool": "stata_run", "args": {"code": code}}

    # 构建给用户看的消息
    tool_display = action["tool"]
    if tool_display == "stata_run":
        display = f"已生成代码:\n```stata\n{code}\n```"
    elif tool_display == "read_file":
        display = f"正在读取文件: {action['args'].get('path', '?')}"
    elif tool_display == "create_do_file":
        display = f"正在创建 .do 文件: {action['args'].get('path', '?')}"
    elif tool_display == "modify_do_file":
        display = f"正在修改 .do 文件: {action['args'].get('path', '?')}"
    else:
        display = f"准备执行: {tool_display}"

    return {
        "messages": [AIMessage(content=display)],
        "current_code": code,
        "error_info": "",
        "retry_count": 0,
        "pending_action": action,
    }


def prepare_tool_call(state: StataAgentState) -> dict:
    """节点2：根据 pending_action 将当前动作封装为工具调用消息"""
    action = state.get("pending_action", {})
    tool_name = action.get("tool", "stata_run")
    args = action.get("args", {"code": state.get("current_code", "")})

    tool_call_msg = AIMessage(
        content="",
        tool_calls=[{
            "id": f"call_{tool_name}",
            "name": tool_name,
            "args": args,
        }]
    )
    return {"messages": [tool_call_msg]}


async def execute_tools(state: StataAgentState, tool_node) -> dict:
    """节点3：执行工具调用，并从结果中提取错误信息写入 state"""
    last_message = state["messages"][-1]
    tool_messages = []
    error_info = ""

    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("args", {})
            tool_id = tool_call.get("id", "unknown")

            try:
                result = await _invoke_tool(tool_node, tool_name, **tool_args)
            except Exception as e:
                result = f"Error: 工具执行失败: {str(e)}"

            result_str = str(result)

            if "error" in result_str.lower() or "r(" in result_str:
                error_info = result_str[:500]

            tool_messages.append(
                ToolMessage(content=result_str, tool_call_id=tool_id)
            )

    return {
        "messages": tool_messages,
        "error_info": error_info,
    }


async def review_error(state: StataAgentState, llm_client) -> dict:
    """节点4：根据错误信息反思并修正代码"""
    error_info = state.get("error_info", "")
    original_code = state.get("current_code", "")
    retry_count = state.get("retry_count", 0)

    if not error_info or retry_count >= 3:
        return {"error_info": error_info, "retry_count": retry_count}

    user_request = ""
    for msg in state["messages"]:
        if hasattr(msg, 'type') and msg.type == "human":
            user_request = msg.content
            break

    prompt = REVIEWER_SYSTEM_PROMPT.format(
        user_request=user_request,
        error_info=error_info,
        original_code=original_code,
    )
    response = await llm_client.call_reviewer([{"role": "system", "content": prompt}])
    corrected_code = extract_code(response)

    return {
        "messages": [AIMessage(content=f"已修正代码:\n```stata\n{corrected_code}\n```")],
        "current_code": corrected_code,
        "error_info": "",
        "retry_count": retry_count + 1,
        "pending_action": {"tool": "stata_run", "args": {"code": corrected_code}},
    }


# ==================== 测试 ====================
if __name__ == "__main__":
    import asyncio
    from unittest.mock import AsyncMock, Mock
    from langchain_core.messages import HumanMessage

    async def test():
        # ============ 测试 parse_action_directive ============
        # 无指令
        assert parse_action_directive("```stata\nreg price mpg\n```") is None
        print("✅ parse 无指令 → None")

        # read_file
        result = parse_action_directive("[ACTION:read_file]\npath: test.do\n")
        assert result == {"tool": "read_file", "args": {"path": "test.do"}}
        print("✅ parse read_file")

        # create_do_file with code
        result = parse_action_directive(
            "[ACTION:create_do_file]\npath: out.do\n```stata\nreg price mpg\n```"
        )
        assert result["tool"] == "create_do_file"
        assert result["args"]["path"] == "out.do"
        assert "reg price mpg" in result["args"]["content"]
        print("✅ parse create_do_file with code")

        # modify_do_file
        result = parse_action_directive(
            "[ACTION:modify_do_file]\npath: old.do\n```stata\nnew code\n```"
        )
        assert result["tool"] == "modify_do_file"
        assert result["args"]["path"] == "old.do"
        print("✅ parse modify_do_file")

        # --- 准备 Mock ---
        mock_tool = Mock()
        mock_tool.ainvoke = AsyncMock(return_value='{"ok": true, "results": {"r": {"scalars": {"N": 74}}}}')
        mock_tool_node = Mock()
        mock_tool_node.tools_by_name = {"stata_run": mock_tool}

        mock_llm = Mock()
        mock_llm.call_coder = AsyncMock(return_value="```stata\nsysuse auto, clear\nreg price mpg\n```")
        mock_llm.call_reviewer = AsyncMock(return_value="```stata\nsysuse auto, clear\nreg price mpg\n```")

        # ============ 测试 generate_code（默认 stata_run）============
        state = {
            "messages": [HumanMessage(content="用 mpg 预测 price")],
            "current_code": "",
            "error_info": "",
            "retry_count": 0,
            "pending_action": {},
        }
        result = await generate_code(state, mock_llm)
        assert result["pending_action"]["tool"] == "stata_run"
        assert "sysuse auto" in result["pending_action"]["args"]["code"]
        print("✅ generate_code → 默认 stata_run")

        # ============ 测试 generate_code（read_file 指令）============
        mock_llm.call_coder = AsyncMock(return_value="[ACTION:read_file]\npath: analysis.do\n")
        result = await generate_code(state, mock_llm)
        assert result["pending_action"]["tool"] == "read_file"
        assert result["pending_action"]["args"]["path"] == "analysis.do"
        print("✅ generate_code → 识别 read_file 指令")

        # ============ 测试 generate_code（create_do_file 指令）============
        mock_llm.call_coder = AsyncMock(return_value="[ACTION:create_do_file]\npath: output.do\n```stata\nreg price mpg\n```")
        result = await generate_code(state, mock_llm)
        assert result["pending_action"]["tool"] == "create_do_file"
        assert result["pending_action"]["args"]["path"] == "output.do"
        assert "reg price mpg" in result["pending_action"]["args"]["content"]
        print("✅ generate_code → 识别 create_do_file 指令")

        # ============ 测试 prepare_tool_call (stata_run) ============
        state["pending_action"] = {"tool": "stata_run", "args": {"code": "reg price mpg"}}
        result = prepare_tool_call(state)
        assert result["messages"][0].tool_calls[0]["name"] == "stata_run"
        assert result["messages"][0].tool_calls[0]["args"]["code"] == "reg price mpg"
        print("✅ prepare_tool_call → stata_run")

        # ============ 测试 prepare_tool_call (read_file) ============
        state["pending_action"] = {"tool": "read_file", "args": {"path": "test.do"}}
        result = prepare_tool_call(state)
        assert result["messages"][0].tool_calls[0]["name"] == "read_file"
        assert result["messages"][0].tool_calls[0]["args"]["path"] == "test.do"
        print("✅ prepare_tool_call → read_file")

        # ============ 测试 prepare_tool_call (create_do_file) ============
        state["pending_action"] = {
            "tool": "create_do_file",
            "args": {"path": "out.do", "content": "reg price mpg"}
        }
        result = prepare_tool_call(state)
        assert result["messages"][0].tool_calls[0]["name"] == "create_do_file"
        assert result["messages"][0].tool_calls[0]["args"]["path"] == "out.do"
        assert result["messages"][0].tool_calls[0]["args"]["content"] == "reg price mpg"
        print("✅ prepare_tool_call → create_do_file")

        print("\n🎉 nodes 全部测试通过 (12 tests)")

    asyncio.run(test())
