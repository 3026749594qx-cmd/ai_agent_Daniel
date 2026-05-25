# ==================== 第一部分：导入依赖 ====================
import json
import re
import os
import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI
from web_search_china import WebSearchChina

# ==================== 第二部分：配置加载 ====================
# 从项目同目录的 config.json 中读取智谱 API Key，写入环境变量
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
if os.path.exists(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    os.environ.setdefault("ZHIPU_API_KEY", config.get("ZHIPU_API_KEY", ""))

# ==================== 第三部分：系统提示词（全局常量） ====================
SYSTEM_PROMPT = """你是一个轻量化的 AI Agent，能够调用以下工具来完成用户用自然语言提出的任务。

【可用工具清单】

1. search_web(query: str) -> str
   使用国内搜索引擎搜索网络信息，返回摘要和链接。适合查人物、概念、新闻等。

2. calculate(expression: str) -> str
   计算数学表达式。例如 "175 + 3" 或 "(100-20)*3"。

3. plot_height_distribution(input: str) -> str
   根据身高和姓名绘制全球身高分布图。
   输入格式："身高(cm), 姓名"，例如 "226.0, 姚明"。
   返回图片保存路径。

4. write_file(input: str) -> str
   写入本地文件（仅限当前目录）。
   输入格式："文件名, 内容"。
   例如 "data.txt, 姚明身高226cm"。

5. read_file(filename: str) -> str
   读取本地文件内容（仅限当前目录）。

【你的能力边界】
- ✅ 查信息、算数、画身高分布图、读写当前目录下的文件
- ❌ 超出以上工具范围的任务无法完成，请诚实告知用户

【工作规则】
- 每次只调用一个工具
- 需要调用工具时，严格输出 JSON：
{"thought": "思考过程", "action": "工具名", "action_input": "参数"}
- 任务完成时输出：
{"thought": "总结", "final_answer": "给用户的回答"}
- 只输出 JSON，不要有其他文字
"""

# ==================== 第四部分：基础安全函数 ====================
# 项目根目录（允许操作的文件路径上限）
ALLOWED_ROOT = os.path.abspath(".")

def safe_path(filename: str) -> str:
    """将传入的相对或绝对路径规范化，并检查是否在 ALLOWED_ROOT 以内"""
    abs_path = os.path.abspath(filename)
    if os.path.commonpath([abs_path, ALLOWED_ROOT]) != ALLOWED_ROOT:
        raise PermissionError(f"不允许操作目录外的文件: {filename}")
    return abs_path

# ==================== 第五部分：五个工具函数 ====================
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

    # 设置中文字体，避免乱码
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
    plt.rcParams['axes.unicode_minus'] = False

    # 绘制正态分布曲线
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

# ==================== 第六部分：工具注册表 ====================
TOOLS = {
    "search_web": search_web,
    "calculate": calculate,
    "plot_height_distribution": plot_height_distribution,
    "write_file": write_file,
    "read_file": read_file,
}

# ==================== 第七部分：Agent 核心循环 ====================
def process_query(client, model, messages, max_steps=8):
    """执行 ReAct 循环，接收消息历史，返回最终回答"""
    for step in range(max_steps):
        # 1. 调用大模型，获取响应文本
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0
        )
        raw = resp.choices[0].message.content.strip()
        print(f"\n[Step {step+1}] 模型输出: {raw}")

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

        # 3. 判断是否为最终回答
        if "final_answer" in action_json:
            return action_json["final_answer"]

        # 4. 处理工具调用
        if "action" in action_json and "action_input" in action_json:
            tool_name = action_json["action"]
            tool_input = action_json["action_input"]
            print(f">>> 调用工具: {tool_name}({tool_input})")

            func = TOOLS.get(tool_name)
            if not func:
                observation = f"工具 '{tool_name}' 不存在。"
            else:
                try:
                    observation = func(tool_input)
                except Exception as e:
                    observation = f"执行错误: {e}"

            print(f">>> 观察结果: {str(observation)[:200]}")
            # 将模型输出和观察结果追加回消息历史
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"工具执行结果: {observation}"})
        else:
            return "模型输出格式错误，请重试。"

    return "任务步数超限，请简化需求后重试。"

# ==================== 第八部分：主交互函数 ====================
def main():
    # 获取 API Key
    api_key = os.getenv("ZHIPU_API_KEY")
    if not api_key:
        print("❌ 未配置 API Key！请确保 config.json 存在于 agent.py 同目录，且格式为：")
        print('   {"ZHIPU_API_KEY": "你的Key"}')
        return

    # 初始化客户端和模型
    client = OpenAI(
        api_key=api_key,
        base_url="https://open.bigmodel.cn/api/paas/v4/"
    )
    model = "glm-4.7-flash"

    # 打印欢迎界面
    print("=" * 60)
    print("🤖 轻量化 AI Agent 已启动")
    print("=" * 60)
    print("【已配置工具】")
    print("  1. search_web    - bing搜索")
    print("  2. calculate     - 数学计算")
    print("  3. plot_height_distribution - 身高分布图")
    print("  4. write_file    - 写入本地文件（当前目录）")
    print("  5. read_file     - 读取本地文件（当前目录）")
    print("【能力范围】查信息 / 算数 / 画身高分布图 / 读写当前目录文件")
    print("【输入 'exit' 或 'quit' 退出】")
    print("=" * 60)

    # 初始化消息历史（第一条是系统提示词）
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # 主交互循环
    while True:
        user_input = input("\n🧑 你好！命令我干活吧: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("👋 Agent 已退出。")
            break
        if user_input.lower() == "help":
            print("可用工具: search_web, calculate, plot_height_distribution, write_file, read_file")
            print("能力范围: 查信息 / 算数 / 画身高分布图 / 读写当前目录文件")
            continue

        # 将用户输入加入消息历史
        messages.append({"role": "user", "content": user_input})

        # 调用核心循环处理
        print("\n--- 开始处理 ---")
        final = process_query(client, model, messages)
        print(f"\n🤖 Agent: {final}")

        # 将最终答案加入消息历史，保持多轮对话
        messages.append({"role": "assistant", "content": final})

# ==================== 第九部分：程序入口 ====================
if __name__ == "__main__":
    main()