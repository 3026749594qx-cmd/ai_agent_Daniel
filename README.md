# 🤖 轻量化 AI Agent 框架

## 介绍
《人工智能基础A》课程大作业——AI Agent 框架搭建。
一个从零搭建的轻量级 AI Agent，支持自然语言交互、多工具调用、流式输出、长期记忆以及多 Agent 协作。


## 特性

1. **ReAct 范式**：交替进行思考（Reasoning）与行动（Acting）
2. **7 个内置工具**：网页搜索、全文提取、数学计算、身高分布图绘制、文件读写、长期记忆
3. **流式输出**：模拟打字效果，提升交互体验
4. **长期记忆**：基于 JSON 文件存储用户偏好，跨会话持久化
5. **多 Agent 协作**：输入“协作：”前缀自动拆分复杂任务并逐步执行
6. **模型配置外部化**：支持任意 OpenAI 兼容 API（智谱/DeepSeek/OpenAI 等）
7. **文件安全沙箱**：文件操作仅限当前工作目录

## 环境准备

### 1. 环境配置
#### 方法一：Conda 环境（推荐）
创建独立隔离环境，避免版本冲突

conda create -n ai_agent python=3.10 -y
conda activate ai_agent

#### 方法二：纯 Python 环境（无需 Conda）
仅需电脑安装 **Python 3.10** 解释器，直接运行项目即可

### 2. 安装依赖
pip install -r requirements.txt

### 3. 安装 Playwright 浏览器内核
playwright install chromium

## 配置 API

1. 复制配置模板：
cp config.example.json config.json
2. 编辑 `config.json`，填入你的 API 信息：
{
  "api_key": "",
  "base_url": "",
  "model_name": ""
}

## 运行
(若使用Conda 环境）conda activate ai_agent

python agent.py

## 使用示例

### 普通模式
🧑 你好！命令我干活吧: 帮我查姚明的身高，画个身高分布图，并把数据存成文档

### 协作模式
🧑 你好！命令我干活吧: 协作：搜索2024年诺贝尔物理学奖得主，提取其主要贡献，然后保存为文件

### 长期记忆
🧑: 记住，我喜欢蓝色
下次启动后：你知道我喜欢什么颜色吗？

## 项目结构
ai_agent_Daniel/
├── agent.py                 # 主程序
├── web_search_china.py      # 搜索引擎封装
├── config.example.json      # 配置模板
├── requirements.txt         # 依赖清单
├── .gitignore               # 忽略规则
├── memory.json              # 长期记忆
├── products/                # 自动生成的输出文件夹（不提交）
│   ├── *.png                # 身高分布图等图片
│   └── *.txt                # 分析报告等文档
└── README.md                # 项目说明

## 进阶功能
- **多 Agent 协作**：通过规划 Agent 拆分任务，执行 Agent 逐步完成
- **全文提取**：使用 Trafilatura 从搜索结果中提取完整正文
- **流式输出**：逐字打印，模拟真实对话感
- **工具扩展**：可轻松添加新工具，只需遵循 ReAct JSON 格式

## 注意事项
- 请勿提交 `config.json`（已在 .gitignore 中忽略）
- 首次运行会自动生成 `memory.json` 存储长期记忆
- 绘图中文显示依赖系统字体（Windows 已自带黑体）

## 许可证
本项目仅用于教育目的。