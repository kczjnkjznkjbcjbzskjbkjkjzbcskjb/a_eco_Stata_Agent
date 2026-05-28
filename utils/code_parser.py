"""
代码解析工具
从 LLM 响应中提取 Stata 代码块
"""

def extract_code(text: str) -> str:
    """从 LLM 响应中提取 ```stata ... ``` 或 ``` ... ``` 代码块"""
    if "```stata" in text:
        start = text.find("```stata") + len("```stata")
        end = text.find("```", start)
        if end != -1:
            return text[start:end].strip()
    if "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end != -1:
            return text[start:end].strip()
    return text.strip()


# ==================== 测试 ====================
if __name__ == "__main__":
    # 测试 stata 代码块
    text1 = "下面是代码：\n```stata\nsysuse auto, clear\nreg price mpg\n```\n请执行。"
    assert extract_code(text1) == "sysuse auto, clear\nreg price mpg"

    # 测试普通代码块
    text2 = "```\nsysuse auto\n```"
    assert extract_code(text2) == "sysuse auto"

    # 测试无代码块
    text3 = "sysuse auto, clear"
    assert extract_code(text3) == "sysuse auto, clear"

    print("✅ code_parser 全部测试通过")