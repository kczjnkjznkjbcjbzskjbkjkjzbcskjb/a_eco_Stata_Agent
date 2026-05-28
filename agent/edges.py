"""
条件边逻辑
决定工具执行后是进入反思节点、重新生成还是结束
"""
from typing import Literal
from langgraph.graph import END
from langchain_core.messages import ToolMessage
from agent.state import StataAgentState
from config.settings import MAX_RETRY


def should_continue(state: StataAgentState) -> Literal["review", "generate", "__end__"]:
    """检查工具执行结果，决定下一步路由

    - "review"   : 有 Stata 错误且未超过重试次数 → 进入修正循环
    - "generate" : read_file 成功 → 回到 generate 让 LLM 基于文件内容做下一步
    - "__end__"   : stata_run 成功 / create_do_file 完成 / modify_do_file 完成 / 超重试 → 结束
    """
    messages = state["messages"]
    retry_count = state.get("retry_count", 0)
    pending_action = state.get("pending_action", {})

    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            content = str(msg.content)
            content_lower = content.lower()

            has_stata_error = "r(" in content_lower
            has_general_error = "error" in content_lower

            # Stata 代码执行错误 → 进入修正循环
            if has_stata_error and retry_count < MAX_RETRY:
                return "review"

            # 超过重试次数 或 文件工具错误（非 Stata 错误，无法修正） → 结束
            if has_stata_error or has_general_error:
                return END

            # read_file 成功 → 链回 generate（LLM 看到文件内容后决定下一步）
            if "读取文件成功" in content_lower:
                return "generate"

            # create_do_file / modify_do_file 成功 → 终态结束
            if "文件已创建" in content_lower or "文件已修改" in content_lower:
                return END

            # stata_run 成功 → 结束
            return END

    return END


# 兼容旧名称
should_retry = should_continue


# ==================== 测试 ====================
if __name__ == "__main__":
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

    # Stata 错误 + 未超重试 → review
    state1 = {
        "messages": [
            HumanMessage(content="做回归"),
            AIMessage(content="生成代码"),
            ToolMessage(content="Error: variable mpg not found r(111)", tool_call_id="c1"),
        ],
        "current_code": "reg price mpg",
        "error_info": "variable mpg not found r(111)",
        "retry_count": 0,
        "pending_action": {"tool": "stata_run", "args": {}},
    }
    assert should_continue(state1) == "review"
    print("✅ Stata r( 错误 + 未超重试 → review")

    # stata_run 成功 → END
    state2 = {
        "messages": [
            HumanMessage(content="做回归"),
            AIMessage(content="生成代码"),
            ToolMessage(content='{"ok": true, "results": {"r": {"scalars": {"N": 74}}}}', tool_call_id="c1"),
        ],
        "current_code": "reg price mpg",
        "error_info": "",
        "retry_count": 0,
        "pending_action": {"tool": "stata_run", "args": {}},
    }
    assert should_continue(state2) == END
    print("✅ stata_run 成功 → END")

    # Stata 错误 + 超重试 → END
    state3 = {
        "messages": [
            HumanMessage(content="做回归"),
            AIMessage(content="生成代码"),
            ToolMessage(content="Error: r(198)", tool_call_id="c1"),
        ],
        "current_code": "reg price mpg",
        "error_info": "r(198)",
        "retry_count": 3,
        "pending_action": {"tool": "stata_run", "args": {}},
    }
    assert should_continue(state3) == END
    print("✅ Stata r( 错误 + 超重试 → END")

    # Stata 错误码 r( → review
    state4 = {
        "messages": [
            HumanMessage(content="做回归"),
            ToolMessage(content='{"ok": false, "rc": -1, "log": {"tail": "r(111)"}}', tool_call_id="c1"),
        ],
        "current_code": "reg price mpg",
        "error_info": "r(111)",
        "retry_count": 0,
        "pending_action": {"tool": "stata_run", "args": {}},
    }
    assert should_continue(state4) == "review"
    print("✅ Stata r( 错误码 → review")

    # read_file 成功 → generate（链式：继续基于内容执行下一步）
    state5 = {
        "messages": [
            HumanMessage(content="读取 analysis.do 并运行"),
            AIMessage(content="正在读取文件"),
            ToolMessage(content="读取文件成功: analysis.do (120 字符, 编码: utf-8)\nsysuse auto, clear\nreg price mpg", tool_call_id="c2"),
        ],
        "current_code": "",
        "error_info": "",
        "retry_count": 0,
        "pending_action": {"tool": "read_file", "args": {"path": "analysis.do"}},
    }
    assert should_continue(state5) == "generate"
    print("✅ read_file 成功 → generate（链式）")

    # create_do_file 成功 → END（终态，不需要再做什么）
    state6 = {
        "messages": [
            HumanMessage(content="保存代码到 output.do"),
            AIMessage(content="正在创建文件"),
            ToolMessage(content="文件已创建: output.do\n--- 内容 ---\nreg price mpg", tool_call_id="c3"),
        ],
        "current_code": "reg price mpg",
        "error_info": "",
        "retry_count": 0,
        "pending_action": {"tool": "create_do_file", "args": {"path": "output.do"}},
    }
    assert should_continue(state6) == END
    print("✅ create_do_file 成功 → END（终态）")

    # modify_do_file 成功 → END（终态）
    state7 = {
        "messages": [
            HumanMessage(content="修改 output.do"),
            AIMessage(content="正在修改文件"),
            ToolMessage(content="文件已修改: output.do\n--- 新内容 ---\nnew code", tool_call_id="c4"),
        ],
        "current_code": "new code",
        "error_info": "",
        "retry_count": 0,
        "pending_action": {"tool": "modify_do_file", "args": {"path": "output.do"}},
    }
    assert should_continue(state7) == END
    print("✅ modify_do_file 成功 → END（终态）")

    # 文件工具错误（非 Stata 错误） → END（无法通过 review 修正）
    state8 = {
        "messages": [
            HumanMessage(content="读取不存在的文件"),
            ToolMessage(content="Error: 文件不存在: ghost.do", tool_call_id="c5"),
        ],
        "current_code": "",
        "error_info": "文件不存在",
        "retry_count": 0,
        "pending_action": {"tool": "read_file", "args": {"path": "ghost.do"}},
    }
    assert should_continue(state8) == END
    print("✅ 文件工具错误 → END（不进入 review）")

    print("\n🎉 edges 全部测试通过 (8 tests)")
