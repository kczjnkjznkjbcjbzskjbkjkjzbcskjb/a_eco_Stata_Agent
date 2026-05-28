"""
双模型 LLM 客户端
编程模型：非思考模式，快速生成 Stata 代码
检查模型：思考模式，深度分析错误并修正代码
"""
from openai import AsyncOpenAI
from config.settings import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    CODER_MODEL,
    REVIEWER_MODEL,
    CODER_TEMPERATURE,
    REVIEWER_TEMPERATURE,
)


class DualLLMClient:
    """封装两个 DeepSeek 模型实例，分别用于代码生成和错误审查"""

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )

    async def call_coder(self, messages: list[dict]) -> str:
        """调用编程模型（非思考模式），快速生成 Stata 代码"""
        response = await self.client.chat.completions.create(
            model=CODER_MODEL,
            messages=messages,
            extra_body={"thinking": {"type": "disabled"}},
            temperature=CODER_TEMPERATURE,
        )
        return response.choices[0].message.content

    async def call_reviewer(self, messages: list[dict]) -> str:
        """调用检查模型（思考模式），深度分析错误并返回修正方案"""
        response = await self.client.chat.completions.create(
            model=REVIEWER_MODEL,
            messages=messages,
            extra_body={"thinking": {"type": "enabled"}},
            temperature=REVIEWER_TEMPERATURE,
        )
        return response.choices[0].message.content