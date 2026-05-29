# ==================== 第一部分：导入依赖 ====================
import json
import re
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI
from web_search_china import WebSearchChina
import trafilatura   # 新增：用于全文提取

# ==================== 第二部分：配置加载 ====================
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
if os.path.exists(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    os.environ.setdefault("ZHIPU_API_KEY", config.get("ZHIPU_API_KEY", ""))

# ==================== 第三部分：系统提示词（全局常量） ====================
SYSTEM_PROMPT = """你是一个轻量化的 AI Agent，能够调用以下工具来完成用户用自然语言提出的任务。

【可用工具清单】

1. search_web(query: str) -> str
   使用必应搜索，返回标题、摘要和链接。摘要通常较短，可能不包含完整信息。

2. fetch_full_text(url: str) -> str
   获取指定网页的完整正文内容（纯文本）。输入：完整的 http/https 链接。
   适用场景：当搜索结果中的摘要信息不足以回答用户问题时，使用本工具深入阅读全文。

3. calculate(expression: str) -> str
   计算数学表达式。例如 "175 + 3" 或 "(100-20)*3"。

4. plot_height_distribution(input: str) -> str
   根据身高和姓名绘制全球身高分布图。输入格式："身高(cm), 姓名"。

5. write_file(input: str) -> str
   写入本地文件，输入格式："文件名, 内容"。

6. read_file(filename: str) -> str
   读取本地文件内容。

7. remember(input: str) -> str
   记住一条用户信息。输入格式："键, 值"。

【工作规则 - 非常重要！】

1. 每次只能调用一个工具。

2. 当用户要求搜索信息时，你必须遵循以下“两步走”策略：
   - 第一步：调用 search_web 获取摘要和链接。
   - 第二步：**你必须至少打开一个搜索结果链接获取全文，除非摘要已经明确给出完整答案**。
   - 明确给出完整答案的判断标准：摘要中已经包含了用户所需的所有具体信息（如具体数字、日期、定义等），且没有“...”或“点击阅读全文”等截断提示。
   - 如果摘要不完整或信息量不足，你必须选择最相关的 1 个链接，调用 fetch_full_text 获取全文内容。

3. 你完全自主决定是否需要提取全文，无需用户额外指令。**强制要求：在大多数搜索任务中，你应当主动获取至少一个链接的全文。**

4. 基于全文（或完整摘要）生成最终回答。

【输出格式】
- 调用工具时输出：
{"thought": "思考过程", "action": "工具名", "action_input": "参数"}
- 任务完成时输出：
{"thought": "总结", "final_answer": "给用户的回答"}

【示例对话】

用户：姚明身高多少？
Agent 第一步输出：
{"thought": "需要先搜索姚明身高", "action": "search_web", "action_input": "姚明身高"}
（观察返回：标题：姚明身高_百度百科，摘要：姚明，身高226厘米。链接：...）
此时摘要已明确给出完整答案（226厘米），无需提取全文。
Agent 最终输出：
{"thought": "摘要已包含准确身高", "final_answer": "姚明的身高是226厘米。"}

用户：搜索一下2025年AI的最新突破。
Agent 第一步输出：
{"thought": "先搜索AI新闻", "action": "search_web", "action_input": "2025年AI重大突破"}
（观察返回：标题：AI新模型发布，摘要：该模型在...（此处省略200字）...点击阅读全文。摘要明显被截断）
由于摘要不完整，Agent 必须继续：
Agent 第二步输出：
{"thought": "摘要信息不完整，需要打开第一个链接获取全文", "action": "fetch_full_text", "action_input": "https://news.com/ai-breakthrough"}
（观察返回：完整正文约3000字，详细描述了三个突破点）
Agent 最终输出：
{"thought": "基于全文提取到三个突破点", "final_answer": "根据最新报道，2025年AI有三大突破：1. ... 2. ... 3. ..."}

【你的能力边界】
- ✅ 查信息、深入阅读网页、算数、画身高分布图、读写文件、记住信息
- ❌ 超出以上工具范围的任务无法完成

【工作规则】
- 每次只调用一个工具
- 需要调用工具时，严格输出 JSON：
{"thought": "思考过程", "action": "工具名", "action_input": "参数"}
- 任务完成时输出：
{"thought": "总结", "final_answer": "给用户的回答"}
- 只输出 JSON，不要有其他文字
"""

# ==================== 第四部分：流式输出辅助函数 ====================
def stream_print(text: str, delay: float = 0.02, end: str = "\n"):
    """逐字打印文本，模拟流式输出效果"""
    for ch in text:
        print(ch, end='', flush=True)
        time.sleep(delay)
    print(end, end='', flush=True)

# ==================== 第五部分：基础安全函数 ====================
ALLOWED_ROOT = os.path.abspath(".")

def safe_path(filename: str) -> str:
    """将传入的相对或绝对路径规范化，并检查是否在 ALLOWED_ROOT 以内"""
    abs_path = os.path.abspath(filename)
    if os.path.commonpath([abs_path, ALLOWED_ROOT]) != ALLOWED_ROOT:
        raise PermissionError(f"不允许操作目录外的文件: {filename}")
    return abs_path

# ==================== 第六部分：工具函数（含新增全文提取） ====================
def search_web(query: str) -> str:
    """使用必应搜索，返回摘要与链接"""
    try:
        searcher = WebSearchChina()
        results = searcher.search(query, engine="bing", count=3)
        if not results:
            return "未找到搜索结果，请尝试更换关键词。"
        formatted = []
        for item in results:
            title = item.get('title', '无标题')
            snippet = item.get('snippet', '无摘要')
            url = item.get('url', '无链接')
            formatted.append(f"标题：{title}\n摘要：{snippet}\n链接：{url}")
        return "\n\n".join(formatted)
    except Exception as e:
        return f"搜索时发生错误: {str(e)}"

def fetch_full_text(url: str) -> str:
    """
    使用 Trafilatura 提取网页正文（纯文本）
    参数:
        url: 完整的 http/https 链接
    返回:
        正文文本（最长 4000 字符），若失败则返回错误信息
    """
    # 简单的 URL 合法性检查
    if not (url.startswith("http://") or url.startswith("https://")):
        return "无效的 URL，必须以 http:// 或 https:// 开头。"
    try:
        # 下载页面（内置超时）
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            return f"无法获取页面内容，请检查 URL 是否可访问: {url}"

        # 提取正文（关闭注释和表格保留）
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            include_formatting=True
        )
        if not text:
            return "未能提取到有效正文，页面可能为视频、图片集或无主要内容。"

        # 控制返回长度（避免 token 爆炸）
        max_chars = 4000
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n... (内容过长，已截断)"

        return text.strip()
    except Exception as e:
        return f"提取正文时发生错误: {str(e)}"

def calculate(expression: str) -> str:
    """安全计算数学表达式，仅允许数字和基础运算符"""
    allowed_chars = set("0123456789+-*/().^% ")
    if not all(c in allowed_chars for c in expression):
        return "表达式含非法字符。"
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算出错: {e}"

def plot_height_distribution(input_str: str) -> str:
    """根据身高和姓名绘制全球身高分布图并保存为PNG"""
    parts = [p.strip() for p in input_str.split(",", 1)]
    if len(parts) != 2:
        return "格式错误，应为：身高(cm), 姓名"
    try:
        height_cm = float(parts[0])
    except ValueError:
        return "身高必须是数字。"
    name = parts[1]

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
    plt.rcParams['axes.unicode_minus'] = False

    mean, std = 170.0, 7.0
    x = np.linspace(mean - 4*std, mean + 4*std, 500)
    y = (1/(std*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mean)/std)**2)

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label='Global Height Distribution')
    plt.axvline(height_cm, color='red', linestyle='--', label=f'{name}: {height_cm} cm')
    plt.title(f'{name} in Global Height Distribution')
    plt.xlabel('Height (cm)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(alpha=0.3)

    filename = f"{name}_height_distribution.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    return f"图片已保存: {os.path.abspath(filename)}"

def write_file(input_str: str) -> str:
    """写入文件，输入格式：文件名, 内容"""
    parts = input_str.split(",", 1)
    if len(parts) != 2:
        return "格式错误，应为：文件名, 内容"
    fname, content = parts[0].strip(), parts[1].strip()
    try:
        path = safe_path(fname)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"文件 {fname} 写入成功。"
    except PermissionError as e:
        return f"写入被拒绝: {e}"
    except Exception as e:
        return f"写入失败: {e}"

def read_file(filename: str) -> str:
    """读取文件内容（仅限当前目录）"""
    try:
        path = safe_path(filename.strip())
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except PermissionError as e:
        return f"读取被拒绝: {e}"
    except Exception as e:
        return f"读取失败: {e}"

def remember(input_str: str) -> str:
    """记住用户提供的一条信息。格式："键, 值" """
    parts = input_str.split(",", 1)
    if len(parts) != 2:
        return "格式错误，应为：键, 值。例如：喜欢的颜色, 蓝色"
    key, value = parts[0].strip(), parts[1].strip()
    if not key or not value:
        return "键和值都不能为空。"
    update_user_info(key, value)
    return f"已记住：{key} = {value}"

# ==================== 第七部分：长期记忆管理 ====================
MEMORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory.json")

def load_memory():
    """从本地 JSON 文件加载记忆字典"""
    if not os.path.exists(MEMORY_FILE):
        return {"user_info": {}, "session_history": []}
    with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_memory(memory_dict):
    """保存记忆字典到本地 JSON 文件"""
    with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(memory_dict, f, ensure_ascii=False, indent=2)

def update_user_info(key, value):
    """更新用户偏好信息"""
    memory = load_memory()
    memory["user_info"][key] = value
    save_memory(memory)

def add_session_summary(summary):
    """添加一条会话摘要到历史记录（保留最近 10 条）"""
    memory = load_memory()
    memory["session_history"].append({
        "timestamp": datetime.now().isoformat(),
        "summary": summary
    })
    if len(memory["session_history"]) > 10:
        memory["session_history"] = memory["session_history"][-10:]
    save_memory(memory)

def build_memory_prompt():
    """从记忆库中抽取用户信息，组合成一段附加提示词"""
    memory = load_memory()
    user_info = memory.get("user_info", {})
    if not user_info:
        return ""
    lines = ["\n【已知用户信息】"]
    for k, v in user_info.items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)

# ==================== 第八部分：工具注册表（已包含 fetch_full_text） ====================
TOOLS = {
    "search_web": search_web,
    "fetch_full_text": fetch_full_text,   # 新增
    "calculate": calculate,
    "plot_height_distribution": plot_height_distribution,
    "write_file": write_file,
    "read_file": read_file,
    "remember": remember,
}

# ==================== 第九部分：Agent 核心循环（含流式输出） ====================
def process_query(client, model, messages, max_steps=8):
    """
    执行 ReAct 循环，所有输出（思考、行动、观察）都通过 stream_print 逐字显示。
    返回最终答案字符串（不打印）。
    """
    for step in range(max_steps):
        # 1. 调用大模型，获取完整响应（为了解析 JSON，仍使用非流式）
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0
        )
        raw = resp.choices[0].message.content.strip()

        # 2. 解析 JSON（正则兜底）
        try:
            action_json = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                try:
                    action_json = json.loads(match.group())
                except json.JSONDecodeError:
                    return "模型输出解析失败，请重试。"
            else:
                return "模型输出解析失败，请重试。"

        # 3. 输出思考过程（流式）
        if "thought" in action_json:
            thought = action_json["thought"]
            print()  # 换行
            stream_print("💭 思考: ", delay=0, end="")
            stream_print(thought, delay=0.02, end="\n\n")

        # 4. 判断是否为最终回答
        if "final_answer" in action_json:
            return action_json["final_answer"]

        # 5. 处理工具调用（需打印行动和观察）
        if "action" in action_json and "action_input" in action_json:
            tool_name = action_json["action"]
            tool_input = action_json["action_input"]
            stream_print("🔧 行动: ", delay=0, end="")
            stream_print(f"{tool_name}({tool_input})", delay=0.02, end="\n\n")

            func = TOOLS.get(tool_name)
            if not func:
                observation = f"工具 '{tool_name}' 不存在。"
            else:
                try:
                    observation = func(tool_input)
                except Exception as e:
                    observation = f"执行错误: {e}"

            stream_print("📋 观察: ", delay=0, end="")
            obs_display = observation[:200] + ("..." if len(observation) > 200 else "")
            stream_print(obs_display, delay=0.02, end="\n\n")

            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"工具执行结果: {observation}"})
        else:
            return "模型输出格式错误，请重试。"

    return "任务步数超限，请简化需求后重试。"

# ==================== 第十部分：主交互函数（含记忆加载与退出摘要） ====================
def main():
    api_key = os.getenv("ZHIPU_API_KEY")
    if not api_key:
        print("❌ 未配置 API Key！请确保 config.json 存在于 agent.py 同目录，且格式为：")
        print('   {"ZHIPU_API_KEY": "你的Key"}')
        return

    client = OpenAI(
        api_key=api_key,
        base_url="https://open.bigmodel.cn/api/paas/v4/"
    )
    model = "glm-4.7-flash"

    # 加载长期记忆，附加到系统提示词
    memory_extension = build_memory_prompt()
    full_system_prompt = SYSTEM_PROMPT + memory_extension

    print("=" * 60)
    print("🤖 轻量化 AI Agent 已启动（流式输出 + 长期记忆 + 全文提取）")
    print("=" * 60)
    print("【已配置工具】")
    print("  1. search_web        - bing搜索（返回标题/摘要/链接）")
    print("  2. fetch_full_text   - 提取网页全文（需提供完整URL）")
    print("  3. calculate         - 数学计算")
    print("  4. plot_height_distribution - 身高分布图")
    print("  5. write_file        - 写入本地文件（当前目录）")
    print("  6. read_file         - 读取本地文件（当前目录）")
    print("  7. remember          - 长期记忆（记住用户信息）")
    print("【能力范围】搜索并深入阅读网页 / 计算 / 绘图 / 文件读写 / 长期记忆")
    print("【输入 'exit' 或 'quit' 退出】")
    print("=" * 60)

    messages = [{"role": "system", "content": full_system_prompt}]

    while True:
        user_input = input("\n🧑 你好！命令我干活吧: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            # 退出前生成会话摘要（仅一次 API 调用，避免速率限制）
            if len(messages) > 2:  # 有实质对话才生成
                summary_prompt = "请用一句话总结本次对话的主要内容，不要超过50字。"
                messages.append({"role": "user", "content": summary_prompt})
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=messages[-10:],  # 仅取最近10条消息，节约 token
                        temperature=0.0
                    )
                    summary = resp.choices[0].message.content.strip()
                    add_session_summary(summary)
                    print(f"📝 会话摘要已保存: {summary}")
                except Exception as e:
                    print(f"⚠️ 生成会话摘要失败: {e}")
            print("👋 Agent 已退出。下次启动时将保留我的记忆。")
            break

        if user_input.lower() == "help":
            print("可用工具: search_web, fetch_full_text, calculate, plot_height_distribution, write_file, read_file, remember")
            print("能力范围: 搜索+深入阅读网页 / 计算 / 绘图 / 文件读写 / 长期记忆")
            continue

        messages.append({"role": "user", "content": user_input})

        print("\n--- 开始处理 ---")
        final_answer = process_query(client, model, messages)

        # 流式输出最终答案
        print()
        stream_print("🤖 Agent: ", delay=0, end="")
        stream_print(final_answer, delay=0.02, end="\n\n")

        messages.append({"role": "assistant", "content": final_answer})

# ==================== 第十一部分：程序入口 ====================
if __name__ == "__main__":
    main()