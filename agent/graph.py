"""
图组装与 StataAgent 主类
将节点函数和条件边组装成完整的 LangGraph 工作流
"""
from functools import partial

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage

from agent.state import StataAgentState
from agent.edges import should_continue
from agent.nodes import generate_code, prepare_tool_call, execute_tools, review_error


class StataAgent:
    """Stata 代码撰写 Agent，封装完整的 LangGraph ReAct 工作流

    流程:
        generate ──→ prepare_tools ──→ execute_tools ──→ review (错误修正) ──↻
           ↑                        ↑                      │
           │                        └──────────────────────┘
           │   (文件工具成功后回到 generate，支持链式操作)
           └────────────────────────────────────────────────┘
    """

    def __init__(self, tool_node: ToolNode, llm_client):
        self.tool_node = tool_node
        self.llm = llm_client
        self.graph = self._build_graph()

    def _build_graph(self):
        """构建 LangGraph 工作流图"""
        workflow = StateGraph(StataAgentState)

        workflow.add_node("generate", partial(generate_code, llm_client=self.llm))
        workflow.add_node("prepare_tools", prepare_tool_call)
        workflow.add_node("execute_tools", partial(execute_tools, tool_node=self.tool_node))
        workflow.add_node("review", partial(review_error, llm_client=self.llm))

        workflow.set_entry_point("generate")
        workflow.add_edge("generate", "prepare_tools")
        workflow.add_edge("prepare_tools", "execute_tools")
        workflow.add_conditional_edges(
            "execute_tools",
            should_continue,
            {
                "review": "review",
                "generate": "generate",
                "__end__": END,
            },
        )
        workflow.add_edge("review", "prepare_tools")

        return workflow.compile()

    async def invoke(self, user_query: str) -> dict:
        """运行 Agent，返回包含 messages / current_code / error_info / retry_count 的最终状态"""
        initial_state = {
            "messages": [HumanMessage(content=user_query)],
            "current_code": "",
            "error_info": "",
            "retry_count": 0,
            "pending_action": {},
        }
        return await self.graph.ainvoke(initial_state)

    def get_graph_mermaid(self) -> str:
        """返回 Mermaid 格式的流程图"""
        return self.graph.get_graph().draw_mermaid()
