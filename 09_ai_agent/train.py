"""AI Agent 演示

学习目标：理解 Agent Loop 运行机制、工具注册与调用流程、Function Calling 模式、
以及 Skill（工具组合）的概念。Agent 是在 LLM 和工具/外部系统之间建立决策循环，
让模型能自主决定"何时调用什么工具、如何根据结果调整下一步行为"。

注释标记：【记忆】【产物】【使用】【练习】

运行模式：
  1. python 09_ai_agent/train.py          默认模式：模拟 Agent Loop + 工具演示
  2. python 09_ai_agent/train.py --live   加载本地 LLM 运行真实 Agent（需 GPU/16GB+ RAM）
  3. python 09_ai_agent/train.py --sim    只看模拟 Agent Loop（无模型，秒级完成）
"""

import sys
import os
import argparse
sys.path.append(".")

from common.utils import set_seed, get_device
from model import Agent, Tool, DEFAULT_TOOLS, _calculator, _search_knowledge, _get_current_time


# ============================================================
# 自定义工具示例：学员可参考此模式添加自己的工具
# ============================================================

def _file_reader(filepath: str) -> str:
    """读取文件内容（演示用，限制大小）。"""
    path = os.path.expanduser(filepath)
    if not os.path.exists(path):
        return f"文件不存在: {filepath}"
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb > 1:
        return f"文件过大 ({size_mb:.1f}MB)，超过 1MB 限制。请指定更小的文件。"
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read(3000)  # 最多读 3000 字符


def demo_simulate():
    """演示 1：模拟 Agent Loop（不加载模型，秒级完成）。

    用预设的工具调用序列展示 Agent 的决策循环机制：
    Thought → Action → Observation → Thought → ... → Final Answer。
    这一步不需要 GPU，也不需要下载模型。
    """
    print("=" * 60)
    print("演示 1：模拟 Agent Loop")
    print("=" * 60)

    # 【记忆】Agent Loop 的核心：LLM 在推理和工具调用之间循环，直到信息足够回答问题。
    #   每一步：模型输出 Thought(思考) + Action(工具名) + Action Input(参数)
    #   → 系统执行工具 → 得到 Observation(结果) → 追加到对话历史
    #   → 模型基于新信息继续 Reasoning → 重复，直到输出 Final Answer
    #
    #   这本质是把"一次性回答"变成了"逐步推理 + 外部信息获取"的循环过程。
    agent = Agent(verbose=True)
    agent.register_tools(DEFAULT_TOOLS)

    # 【记忆】模拟器不需要模型：按预设序列调用工具，展示 Agent Loop 的决策流程。
    #   真实运行时，这些决策由 LLM 自主做出（见 demo_live_agent）。
    result = agent.simulate(
        task="计算 15*23 的结果，并查询「监督学习」的定义，用这两个信息回答问题。",
        tool_sequence=[
            {
                "thought": "需要先计算 15 * 23，然后再查监督学习。先算乘法。",
                "action": "calculator",
                "action_input": '{"expression": "15 * 23"}',
            },
            {
                "thought": "计算结果是 345，现在查监督学习的定义。",
                "action": "search_knowledge",
                "action_input": '{"query": "监督学习"}',
            },
            {
                "thought": "我已经有 15*23=345 和「监督学习」的定义，可以给出最终答案。",
                "final_answer": "15 × 23 = 345。监督学习是使用带标签的数据训练模型的方法，"
                                "常见任务包括分类（如垃圾邮件检测）和回归（如房价预测）。",
            },
        ],
    )

    print(f"\n{'='*60}")
    print(f"最终答案: {result['final_answer']}")
    print(f"总步数: {result['total_steps']}")


def demo_tool_execution():
    """演示 2：工具独立调试。

    在接入 Agent 之前，先单独验证每个工具是否正确执行。
    这是 Agent 开发的黄金法则：先调试工具，再调试循环。
    """
    print("\n" + "=" * 60)
    print("演示 2：工具独立调试")
    print("=" * 60)

    test_cases = [
        ("calculator", {"expression": "sqrt(256) + log2(8)"}, "19.0"),
        ("calculator", {"expression": "min(max(1,5), min(3,10))"}, "3"),
        ("search_knowledge", {"query": "强化学习"}, "奖励信号"),
        ("search_knowledge", {"query": "量子计算"}, "未找到"),  # 库外问题 → 未找到
        ("get_current_time", {}, "2026"),
    ]

    for tool_name, args, expected_hint in test_cases:
        tool = DEFAULT_TOOLS[[t.name for t in DEFAULT_TOOLS].index(tool_name)]
        result = tool.execute(args)
        status = "PASS" if expected_hint in result else "CHECK"
        print(f"  [{status}] {tool_name}({args}) → {result[:80]}...")


def demo_custom_tool():
    """演示 3：注册自定义工具。

    Agent 的能力边界由工具决定。添加工具只需：
    1. 编写一个 Python 函数。
    2. 用 Tool 描述它的功能、参数和实现。
    3. agent.register_tool(...) 注册。
    """
    print("\n" + "=" * 60)
    print("演示 3：注册自定义工具")
    print("=" * 60)

    agent = Agent(verbose=False)
    agent.register_tools(DEFAULT_TOOLS)

    # 【使用】注册自定义工具：只需描述清楚功能 + 参数，Agent 就能调用。
    #   工具描述是 prompt 的一部分 —— 描述越精确、越结构化，Agent 越能正确调用。
    file_tool = Tool(
        name="read_file",
        description="读取本地文本文件内容。输入文件路径，返回文件前 3000 个字符。"
                    "适合读取日志、配置文件或代码。",
        parameters="filepath: 文件路径字符串，如 '09_ai_agent/train.py'",
        func=_file_reader,
    )
    agent.register_tool(file_tool)

    print(f"已注册工具 ({len(agent.tools)} 个):")
    for name, tool in agent.tools.items():
        print(f"  - {name}: {tool.description[:80]}...")

    # 直接测试自定义工具
    result = file_tool.execute({"filepath": "09_ai_agent/train.py"})
    print(f"\nread_file('09_ai_agent/train.py') 前 100 字符:")
    print(f"  {result[:100]}...")


def demo_live_agent():
    """演示 4：加载 LLM 运行真实 Agent。

    首次运行会从 HuggingFace 下载 Qwen2.5-1.5B-Instruct（约 3GB）。
    需要 GPU（推荐 4GB+ VRAM）或 CPU（16GB+ RAM，较慢）。
    如果本地资源不足，先用 demo_simulate 理解机制。
    """
    print("\n" + "=" * 60)
    print("演示 4：加载 LLM 运行真实 Agent")
    print("=" * 60)

    set_seed(42)
    device = get_device()
    print(f"设备: {device}")

    agent = Agent(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        max_steps=5,
        verbose=True,
        device=device,
    )
    agent.register_tools(DEFAULT_TOOLS)

    # 【调参】max_steps：每次工具调用算一步，设置上限防止小模型陷入无效循环。
    #   temperature: 0.3 让工具调用更稳定（降低随机性），但设太高可能导致乱调工具。
    #   max_new_tokens: 512 留足空间给模型输出 Thought + Action + Action Input 的完整三段式。

    task = "请帮我计算 256 的平方根，然后查询「Transformer」是什么。"
    print(f"\n任务: {task}")
    print("-" * 40)

    result = agent.run(task)

    print(f"\n{'='*60}")
    print(f"【产物】最终答案: {result['final_answer']}")
    print(f"总步数: {result['total_steps']}")
    for s in result["steps"]:
        if s.observation:
            print(f"  Step {s.step}: {s.action}({s.action_input}) → {s.observation[:80]}...")


def main():
    parser = argparse.ArgumentParser(description="AI Agent 演示")
    parser.add_argument(
        "--live", action="store_true",
        help="加载本地 LLM 运行真实 Agent（需 GPU/16GB+ RAM）",
    )
    parser.add_argument(
        "--sim", action="store_true",
        help="只运行模拟 Agent Loop（无需模型）",
    )
    args = parser.parse_args()

    print("=== AI Agent Demo ===\n")
    print("Agent 的核心：给 LLM 配备工具，让它在推理和工具调用之间循环，")
    print("自主决定何时调用什么工具、如何根据结果调整行为。\n")

    # 【记忆】Agent 的三个核心概念：
    #   Agent Loop：Thought → Action → Observation → Thought → ... → Final Answer
    #   Function Calling：模型输出结构化的函数名和参数，系统执行后返回结果。
    #     本模块用 ReAct（Reasoning + Acting），是 Function Calling 的经典模式之一。
    #   Skill：多个工具的预设编排，定义了"完成某类任务需要哪些工具及调用顺序"。

    if args.sim:
        demo_simulate()
        return

    # 默认演示模式（无模型）：模拟 Agent Loop + 工具演示
    demo_simulate()
    demo_tool_execution()
    demo_custom_tool()

    if args.live:
        demo_live_agent()
    else:
        print("\n" + "=" * 60)
        print("提示: 运行 'python 09_ai_agent/train.py --live' 加载 LLM 运行真实 Agent")
        print("提示: 运行 'python 09_ai_agent/train.py --sim' 只看模拟 Agent Loop")
        print("=" * 60)

    # 【练习】
    #   1. 新增一个 weather(city) 工具（返回模拟天气），注册到 Agent 并测试。
    #   2. 把 08 RAG 的检索功能包装成 Tool，让 Agent 能搜索本地文档。
    #   3. 修改 system prompt，看不同提示对工具选择的影响。
    #   4. 比较 temperature=0.1 与 temperature=0.7 时工具调用的稳定性。
    #   5. 故意注册两个功能重叠的工具（如 search_knowledge + wiki_search），
    #      看 Agent 如何选择。
    #   6. 尝试让 Agent 调用一个会失败的工具，观察其错误恢复能力。


if __name__ == "__main__":
    main()
