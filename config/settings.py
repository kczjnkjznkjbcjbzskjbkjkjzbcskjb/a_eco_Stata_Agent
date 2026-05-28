"""
全局配置
所有可调参数集中管理，修改配置只需改这一个文件
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ==================== DeepSeek API ====================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# ==================== 模型参数 ====================
CODER_MODEL = "deepseek-v4-pro"
REVIEWER_MODEL = "deepseek-v4-pro"
CODER_TEMPERATURE = 0.3
REVIEWER_TEMPERATURE = 0.3

# ==================== Agent 参数 ====================
MAX_RETRY = 3  # 代码修正最大重试次数

# ==================== Stata 参数 ====================
STATA_WORK_DIR = None  # Stata 工作目录，None 表示使用当前目录

