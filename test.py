from llm.client import DualLLMClient

client = DualLLMClient()

# 测试编程模型
response = client.call_coder([
    {"role": "system", "content": "你是 Stata 编程助手，只输出代码。"},
    {"role": "user", "content": "写一段代码，用 auto 数据集做 price 对 mpg 的回归。"}
])
print("=== Coder 响应 ===")
print(response[:200])

# 测试检查模型
response = client.call_reviewer([
    {"role": "system", "content": "你是代码审查专家。"},
    {"role": "user", "content": "下面代码报错 r(111)，请分析原因。"}
])
print("\n=== Reviewer 响应 ===")
print(response[:200])

with open("llm/client.py", "r", encoding="utf-8") as f:
    print(f.read())