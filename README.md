# 轻量化 AI Agent

> 《人工智能基础 A》课程大作业 —— AI Agent 框架搭建  
> 从零搭建的轻量级 AI Agent，支持自然语言交互、多工具调用、流式输出、长期记忆以及多 Agent 协作。

---

##  核心特性

-  **ReAct 范式** – 交替进行推理（Reasoning）与行动（Acting）
-  **7 个内置工具** – 网页搜索、全文提取、计算器、身高分布图、文件读写、长期记忆
-  **流式输出** – 模拟打字效果，提升交互体验
-  **长期记忆** – 基于 JSON 文件存储用户偏好，跨会话持久化
-  **多 Agent 协作** – 输入 `协作：` 前缀自动拆分复杂任务并逐步执行
-  **模型配置外部化** – 支持任意 OpenAI 兼容 API（智谱 / DeepSeek / OpenAI / 本地模型等）
-  **文件安全沙箱** – 所有生成的文件自动集中到 `products/` 目录，且不允许操作目录外文件

---

##  快速开始：在你的电脑上运行本agent

### 1. 获取项目

在终端中执行：

```bash
git clone https://github.com/3026749594qx-cmd/ai_agent_Daniel.git
cd ai_agent_Daniel
```

### 2. 配置 API 密钥

复制并编辑配置文件：

```bash
# Windows
copy config.example.json config.json
```

打开 `config.json`，填入你的 API 信息：

```json
{
  "api_key": "",
  "base_url": "",
  "model_name": ""
}
```

> 注：支持任何兼容 OpenAI SDK 的服务商，比如智谱、DeepSeek、OpenAI 等。

### 3. 安装 Python 环境

根据你的习惯，选择 **A**（Conda）或 **B**（纯 Python）其中一种即可。

#### 方式 A：Conda 环境（推荐）

```bash
conda create -n ai_agent python=3.10 -y
conda activate ai_agent
```

#### 方式 B：纯 Python 虚拟环境（无需 Conda）
仅需电脑安装 **Python 3.10** 解释器

### 4. 安装依赖

```bash
pip install -r requirements.txt
```

### 5. 安装浏览器内核

```bash
playwright install chromium
```

### 6. 启动 Agent

若使用conda，确保终端已激活对应环境（运行conda activate ai_agent）
运行：

```bash
python agent.py
```

---

## 使用示例

### 普通模式
```
🧑 你好！命令我干活吧: 帮我查姚明的身高，画个身高分布图，并把数据存成文档
```

### 协作模式
```
🧑 你好！命令我干活吧: 协作：搜索2024年诺贝尔物理学奖得主，提取其主要贡献，然后保存为文件
```

### 长期记忆
```
🧑 记住，我喜欢蓝色

（下次启动后）
🧑 你知道我喜欢什么颜色吗？
```

---

##  项目结构

| 文件 / 目录 | 说明 |
|-------------|------|
| `agent.py` | 主程序，包含 Agent 框架、ReAct 循环、工具函数、交互入口 |
| `web_search_china.py` | 中文搜索引擎封装（必应 / 百度 / 360） |
| `config.example.json` | API 配置模板，请复制为 `config.json` 后填入真实信息 |
| `requirements.txt` | Python 依赖清单 |
| `.gitignore` | Git 忽略规则，防止敏感文件被提交 |
| `memory.json` | 长期记忆数据（本地自动生成，不提交） |
| `products/` | Agent 生成的所有输出文件（图片、文档等，不提交） |
| `README.md` | 项目说明文档 |

---

##  进阶功能

- **多 Agent 协作** – 通过规划 Agent 拆分任务，执行 Agent 逐步完成
- **全文提取** – 使用 Trafilatura 从搜索结果中提取完整正文
- **流式输出** – 逐字打印，模拟真实对话感
- **工具扩展** – 可轻松添加新工具，只需遵循 ReAct 的 JSON 格式

---

## 注意事项

- ⚠️**不要提交 `config.json`**：该文件已在 `.gitignore` 中忽略，避免泄露 API 密钥。
- 首次运行会自动生成 `memory.json` 存储长期记忆。
- 绘图中文显示依赖系统字体，Windows 自带黑体一般无需额外配置。

---

## 许可证

本项目仅用于教育目的。