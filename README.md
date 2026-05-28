# a_eco_Stata_Agent

基于 **LangGraph + MCP + DeepSeek** 的 Stata 代码撰写 Agent，支持自动生成 Stata 代码、执行、错误修正，以及文件操作（读取/创建/修改 .do 文件）。

## 架构

```
用户输入 → generate → prepare_tools → execute_tools → review ↻
              ↑            ↑               │
              │            └───────────────┘ (错误修正)
              │   (read_file 成功 → 链回)
              └────────────────────────────┘
                     (写文件成功 → END)
```

## 目录结构

```
a_eco_Stata_Agent/
├── agent/                  # Agent 核心
│   ├── state.py            # 状态定义 (StataAgentState)
│   ├── prompts.py          # 提示词模板
│   ├── nodes.py            # 图节点函数 (generate / prepare / execute / review)
│   ├── edges.py            # 条件边逻辑 (should_continue)
│   ├── graph.py            # 图组装 + StataAgent 主类
│   └── __init__.py
├── tools/                  # 工具模块
│   ├── MCP_stata.py        # Stata MCP 连接管理器
│   ├── stata_tools.py      # MCP 工具 → LangChain 工具适配
│   ├── file_tools.py       # 文件操作工具 (read / create / modify .do)
│   └── __init__.py
├── llm/                    # LLM 客户端
│   ├── client.py           # DualLLMClient (DeepSeek API)
│   └── __init__.py
├── utils/                  # 工具函数
│   ├── code_parser.py      # 从 LLM 输出中提取代码块
│   └── __init__.py
├── config/                 # 配置
│   ├── settings.py         # API Key / 模型参数 / 重试次数
│   └── __init__.py
├── main.py                 # 交互式入口
├── test.py                 # 快速测试脚本
├── study.ipynb             # 开发探索笔记
├── requirements.txt        # 依赖列表
└── .env                    # 环境变量 (API Key)
```

## 工作流

| 步骤 | 节点 | 说明 |
|---|---|---|
| 1 | `generate` | LLM 根据用户需求生成 Stata 代码，或决定文件操作 |
| 2 | `prepare_tools` | 生成代码 → `stata_run`；文件操作 → `read_file` / `create_do_file` / `modify_do_file` |
| 3 | `execute_tools` | 执行工具调用，从结果提取错误信息 |
| 4 | `review` | Stata 代码出错时，LLM 分析错误并修正代码，然后回到步骤 2 重试 |

**退出逻辑：**
- `stata_run` 成功 → END
- `create_do_file` / `modify_do_file` 成功 → END
- `read_file` 成功 → 链回 `generate`（LLM 基于文件内容做下一步）
- Stata 错误 + 未超重试 → `review` 修正循环
- 超重试 / 文件工具错误 → END

## 可用工具

| 工具 | 来源 | 功能 |
|---|---|---|
| `stata_run` | MCP | 执行 Stata 代码并返回结果 |
| `get_log` | MCP | 获取 Stata 运行日志 |
| `read_file` | 本地 | 读取文件内容（自动尝试 utf-8/gbk/gb2312/latin-1） |
| `create_do_file` | 本地 | 创建新的 .do 文件 |
| `modify_do_file` | 本地 | 覆盖已存在的 .do 文件 |

## 快速开始

### 环境要求

- Python 3.11+
- Stata 17+（需安装 pystata）
- DeepSeek API Key
- `stata-code-mcp` 命令可用

### 安装

```bash
cd a_eco_Stata_Agent
pip install -r requirements.txt
```

配置 `.env` 文件：
```
DEEPSEEK_API_KEY=your_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

### 运行

```bash
python main.py
```

交互示例：
```
👤 请输入需求（输入 exit 退出）:
> 用 auto 数据集，以 price 为因变量，mpg 和 weight 为自变量做回归分析

👤 请输入需求:
> 把这段代码保存到 regression.do

👤 请输入需求:
> 读取 regression.do 并执行
```

### 运行测试

```bash
python -m agent.state
python -m agent.prompts
python -m agent.nodes
python -m agent.edges
python -m tools.file_tools
python -m utils.code_parser
```

## 配置说明

`config/settings.py`：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `CODER_MODEL` | `deepseek-v4-pro` | 代码生成模型 |
| `REVIEWER_MODEL` | `deepseek-v4-pro` | 代码审查模型 |
| `CODER_TEMPERATURE` | `0.3` | 生成温度 |
| `REVIEWER_TEMPERATURE` | `0.3` | 审查温度 |
| `MAX_RETRY` | `3` | 代码修正最大重试次数 |

## 依赖

- `openai` — DeepSeek API 调用
- `langgraph` — Agent 工作流编排
- `langchain-core` + `langchain-mcp-adapters` — MCP 工具集成
- `mcp` — MCP 协议客户端
- `python-dotenv` — 环境变量管理
- `pystata` — Stata Python 接口（由 Stata 17 提供）
