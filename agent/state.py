"""
Agent 状态定义
定义在图节点间流转的数据结构
"""
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class StataAgentState(TypedDict):
    """Agent 全局状态，各节点通过读写此状态来传递信息"""
    messages: Annotated[list, add_messages]  # 对话历史（自动追加）
    current_code: str                         # 当前生成的 Stata 代码
    error_info: str                           # 最近一次执行的错误信息
    retry_count: int                          # 当前已重试次数
    pending_action: dict                      # LLM 决定的下一步动作 {"tool": "stata_run|read_file|...", "args": {...}}


# ==================== 测试 ====================
if __name__ == "__main__":
    from langchain_core.messages import HumanMessage, AIMessage

    state: StataAgentState = {
        "messages": [HumanMessage(content="做个回归分析")],
        "current_code": "sysuse auto, clear\nreg price mpg",
        "error_info": "",
        "retry_count": 0,
    }

    # 测试 add_messages 自动追加
    new_state = {"messages": [AIMessage(content="收到")]}
    merged = add_messages(state["messages"], new_state["messages"])
    print(f"消息数: {len(merged)}")
    print(f"第一条: {merged[0].content}")
    print(f"第二条: {merged[1].content}")
    print("✅ state 测试通过")