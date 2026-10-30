"""AI Agent 系统

Agent 的核心是在模型推理和工具调用之间建立循环：
  模型思考 → 决定调用工具 → 执行工具 → 获得结果 → 继续思考 → ... → 最终回答

本模块实现一个轻量 ReAct Agent，教学重点是理解 Agent Loop 运行机制、工具描述
如何影响模型行为，以及 Function Calling 的流程，而非复刻生产级 Agent 框架。
"""

from __future__ import annotations

import json
import re
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Tool：可调用工具的封装
# ---------------------------------------------------------------------------

@dataclass
class Tool:
    """一个可供 Agent 调用的工具。

    name        工具名称，模型通过它决定调用哪个工具
    description 工具功能的自然语言描述（会进入 prompt，越精确模型越会用）
    parameters  参数说明，告诉模型传哪些参数及含义
    func        实际执行的 Python 函数
    """

    name: str
    description: str
    parameters: str  # 简短描述，会出现在 system prompt 中
    func: Callable

    def execute(self, args: Dict[str, Any]) -> str:
        """执行工具并返回结果字符串。"""
        try:
            result = self.func(**args)
            return str(result)
        except Exception as e:
            return f"工具执行错误: {e}"


# ---------------------------------------------------------------------------
# 内置示例工具
# ---------------------------------------------------------------------------

def _calculator(expression: str) -> str:
    """安全计算数学表达式。

    只允许数字、运算符、括号和 math 中的安全函数，禁止任意代码执行。
    """
    allowed = {"__builtins__": {}}
    safe_math = {
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "sqrt": math.sqrt, "log": math.log, "log2": math.log2,
        "log10": math.log10, "pow": math.pow, "pi": math.pi, "e": math.e,
        "abs": abs, "round": round, "min": min, "max": max,
        "sum": sum, "int": int, "float": float,
    }
    allowed.update(safe_math)
    try:
        result = eval(expression, allowed, {})
        return f"{result}"
    except Exception as e:
        return f"计算失败: {e}"


def _search_knowledge(query: str) -> str:
    """模拟知识库检索（教学版：返回预设内容）。

    生产环境中，这里应接入真正的向量检索或搜索引擎。
    在完成 08 RAG 后，学员可将此工具替换为 Chroma 检索。
    """
    knowledge = {
        "监督学习": "监督学习使用带标签的数据训练模型，常见任务包括分类和回归。"
                   "例如：垃圾邮件检测（分类）、房价预测（回归）。",
        "无监督学习": "无监督学习从无标签数据中发现隐藏结构，常见方法：K-Means 聚类、"
                       "PCA 降维、自编码器。",
        "强化学习": "强化学习通过与环境交互获得奖励信号来学习最优策略。"
                    "核心概念：状态(State)、动作(Action)、奖励(Reward)、策略(Policy)。",
        "transformer": "Transformer 是 2017 年提出的序列建模架构，基于自注意力机制。"
                       "它是 GPT、BERT 等现代大模型的基础。",
        "rag": "RAG（检索增强生成）在回答问题前先检索相关文档，再将文档交 LLM 生成答案。"
               "适合需要引用私有知识或频繁更新知识的场景。",
        "lora": "LoRA（Low-Rank Adaptation）是一种参数高效微调方法，只训练低秩矩阵 A·B，"
                "冻结原模型全部参数。可训练参数通常不到全量的 1%。",
    }
    query_lower = query.lower().strip()
    for key, value in knowledge.items():
        if key in query_lower:
            return value
    return f"未找到与「{query}」相关的知识。请换一个关键词或换个角度提问。"


def _get_current_time(_format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """返回当前时间。"""
    return datetime.now().strftime(_format)


# 默认工具集
DEFAULT_TOOLS = [
    Tool(
        name="calculator",
        description="执行数学计算。输入一个数学表达式字符串，返回计算结果。"
                    "支持 + - * / ** 及 sqrt/sin/cos/log 等函数。",
        parameters="expression: 数学表达式字符串，如 '2 + 3 * 4' 或 'sqrt(16)'",
        func=_calculator,
    ),
    Tool(
        name="search_knowledge",
        description="在知识库中搜索概念定义。输入查询关键词，返回相关知识条目。"
                    "适合查询机器学习、深度学习、AI 等领域的专业术语。",
        parameters="query: 要查询的关键词，如 '监督学习' 或 'Transformer'",
        func=_search_knowledge,
    ),
    Tool(
        name="get_current_time",
        description="获取当前日期和时间。不需要参数时传空对象。",
        parameters="_format: 时间格式字符串，默认 '%Y-%m-%d %H:%M:%S'",
        func=_get_current_time,
    ),
]


# ---------------------------------------------------------------------------
# ReAct Agent
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT = """你是一个能调用工具的 AI 助手。你可以使用以下工具完成任务：

{tool_descriptions}

## 工作方式

请按以下格式响应（每次只能执行一个工具调用）：

Thought: 思考当前需要做什么，分析已有信息是否足够回答问题。
Action: 工具名称
Action Input: {{"参数名": "参数值"}}

工具执行后，你会收到 Observation（工具返回结果），然后继续 Thought 循环。

当你认为信息足够回答用户问题时，用以下格式给出最终答案：

Thought: 我已经掌握足够信息，可以给出最终答案。
Final Answer: 你的最终回答

## 重要规则

1. 每次只调用一个工具。
2. 必须严格使用上述 JSON 格式输出 Action Input。
3. 如果工具执行失败或未找到相关信息，尝试换个方式或坦诚说不知道。
4. 回答应基于 Observation 中的实际信息，不能编造。"""

MAX_LOOP_STEPS = 8


@dataclass
class AgentStep:
    """记录 Agent 循环中的每一步。"""
    step: int
    thought: str = ""
    action: str = ""
    action_input: str = ""
    observation: str = ""


class Agent:
    """轻量 ReAct Agent。

    不依赖 LangChain Agent 框架，从零实现 Tool 注册、prompt 构建、
    工具调用解析、执行和循环控制。目的是展示 Agent 的核心机制，
    而不是替代生产级 Agent 框架。

    使用方式:
        agent = Agent(model_name="Qwen/Qwen2.5-1.5B-Instruct")
        agent.register_tools(DEFAULT_TOOLS)
        result = agent.run("计算 15 * 23 并告诉我监督学习的定义")
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        max_steps: int = MAX_LOOP_STEPS,
        device: Optional[str] = None,
        verbose: bool = True,
    ):
        """初始化 Agent。

        Args:
            model_name: HuggingFace 模型名。默认 Qwen2.5-1.5B-Instruct
                        （约 3GB，CPU 可运行）。可换为其他 instruct/chat 模型。
            max_steps: 最大循环步数，防止模型陷入无限工具调用。
            device: 设备字符串。None 时自动选择 CUDA/MPS/CPU。
            verbose: 是否打印每一步的详细过程。
        """
        self.model_name = model_name
        self.max_steps = max_steps
        self.verbose = verbose
        self.tools: Dict[str, Tool] = {}
        self.history: List[AgentStep] = []

        # 延迟加载：模型只在 register_tools 后的 build_prompt 阶段加载，
        # 允许先注册工具再决定是否加载。
        self.model = None
        self.tokenizer = None
        if device is None:
            self.device = self._auto_device()
        else:
            self.device = device

    @staticmethod
    def _auto_device() -> str:
        """自动选择 CUDA / MPS / CPU。"""
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # ---- 工具管理 ----

    def register_tool(self, tool: Tool) -> None:
        """注册单个工具。"""
        self.tools[tool.name] = tool

    def register_tools(self, tools: List[Tool]) -> None:
        """批量注册工具。"""
        for tool in tools:
            self.register_tool(tool)

    def _get_tool_descriptions(self) -> str:
        """构建工具描述文本，用于注入 system prompt。"""
        lines = []
        for tool in self.tools.values():
            lines.append(f"- **{tool.name}**: {tool.description}")
            lines.append(f"  参数: {tool.parameters}")
        return "\n".join(lines)

    # ---- 模型加载 ----

    def _ensure_model_loaded(self):
        """延迟加载模型和 tokenizer（首次 run 时触发）。"""
        if self.model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.verbose:
            print(f"正在加载模型: {self.model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        if self.verbose:
            print(f"模型加载完成，设备: {self.model.device}")

    # ---- Prompt 构建 ----

    def _build_prompt(self, user_task: str, history_steps: List[AgentStep]) -> str:
        """构建完整的对话 prompt。

        包含 system prompt（含工具描述）+ 用户任务 + 历史交互。
        """
        system = AGENT_SYSTEM_PROMPT.format(
            tool_descriptions=self._get_tool_descriptions()
        )

        # Qwen2.5 chat template
        messages = [{"role": "system", "content": system}]
        messages.append({"role": "user", "content": user_task})

        for step in history_steps:
            # 模型上轮的输出
            assistant_msg = f"Thought: {step.thought}\nAction: {step.action}\nAction Input: {step.action_input}"
            messages.append({"role": "assistant", "content": assistant_msg})
            # 工具返回
            messages.append({"role": "user", "content": f"Observation: {step.observation}"})

        self._ensure_model_loaded()
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt

    # ---- 工具调用解析 ----

    @staticmethod
    def _parse_response(text: str) -> Dict[str, str]:
        """从模型输出中解析 Thought、Action、Action Input 和 Final Answer。

        支持两种终止模式：
        - Final Answer: 模型认为信息足够，直接回答。
        - Thought + Action + Action Input: 模型决定调用工具。
        """
        result = {"thought": "", "action": "", "action_input": "", "final_answer": ""}

        # Final Answer 模式
        fa_match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL | re.IGNORECASE)
        if fa_match:
            result["final_answer"] = fa_match.group(1).strip()
            thought_match = re.search(r"Thought:\s*(.+?)(?=Final Answer:)", text, re.DOTALL | re.IGNORECASE)
            if thought_match:
                result["thought"] = thought_match.group(1).strip()
            return result

        # Thought + Action 模式
        thought_match = re.search(r"Thought:\s*(.+?)(?=Action:)", text, re.DOTALL | re.IGNORECASE)
        if thought_match:
            result["thought"] = thought_match.group(1).strip()

        action_match = re.search(r"Action:\s*(\S+)", text)
        if action_match:
            result["action"] = action_match.group(1).strip()

        # Action Input（JSON 格式）
        ai_match = re.search(
            r"Action Input:\s*(\{.+?\})", text, re.DOTALL
        )
        if ai_match:
            result["action_input"] = ai_match.group(1).strip()

        return result

    # ---- 工具执行 ----

    def _execute_tool(self, action_name: str, action_input: str) -> str:
        """执行指定工具并返回 observation。"""
        if action_name not in self.tools:
            available = ", ".join(self.tools.keys())
            return f"未找到工具 '{action_name}'。可用工具: {available}"

        tool = self.tools[action_name]
        try:
            args = json.loads(action_input) if action_input.strip() else {}
        except json.JSONDecodeError:
            return f"Action Input 不是有效的 JSON: {action_input}"

        return tool.execute(args)

    # ---- 主循环 ----

    def run(self, task: str) -> Dict[str, Any]:
        """执行 Agent 主循环。

        Args:
            task: 用户任务描述。

        Returns:
            dict，包含 final_answer、steps、total_steps。
        """
        self.history = []
        final_answer = None

        for step_idx in range(1, self.max_steps + 1):
            # 1. 构建 prompt
            prompt = self._build_prompt(task, self.history)

            # 2. 模型推理
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with self.inference_context():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            # 只取新生成的部分
            generated = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            # 3. 解析输出
            parsed = self._parse_response(generated)

            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Step {step_idx}")
                print(f"{'='*60}")
                print(f"Thought: {parsed['thought']}")
                if parsed["final_answer"]:
                    print(f"Final Answer: {parsed['final_answer']}")
                else:
                    print(f"Action: {parsed['action']}")
                    print(f"Action Input: {parsed['action_input']}")

            # 4. 检查是否给出最终答案
            if parsed["final_answer"]:
                final_answer = parsed["final_answer"]
                # 记录最后一步
                self.history.append(AgentStep(
                    step=step_idx,
                    thought=parsed["thought"],
                    action="",
                    action_input="",
                    observation="",
                ))
                break

            # 5. 执行工具
            observation = self._execute_tool(parsed["action"], parsed["action_input"])
            if self.verbose:
                print(f"Observation: {observation}")

            # 6. 记录步骤
            self.history.append(AgentStep(
                step=step_idx,
                thought=parsed["thought"],
                action=parsed["action"],
                action_input=parsed["action_input"],
                observation=observation,
            ))

        if final_answer is None:
            final_answer = "Agent 在达到最大步数限制后仍未给出最终答案。"

        return {
            "final_answer": final_answer,
            "steps": self.history,
            "total_steps": len(self.history),
        }

    def inference_context(self):
        """返回推理上下文管理器。

        默认使用 torch.no_grad()。
        支持 FP16/BF16 的 autocast 可在此扩展。
        """
        import torch
        return torch.no_grad()

    # ---- 纯文本模拟（无需模型） ----

    def simulate(self, task: str, tool_sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """不加载模型，按预设工具序列模拟 Agent 运行流程。

        用于快速演示 Agent Loop 机制、调试工具逻辑或教学展示。
        """
        self.history = []
        final_answer = None

        for step_idx, planned in enumerate(tool_sequence, 1):
            if "final_answer" in planned:
                final_answer = planned["final_answer"]
                self.history.append(AgentStep(
                    step=step_idx,
                    thought=planned.get("thought", ""),
                    observation="",
                ))
                if self.verbose:
                    print(f"\nStep {step_idx} | Final Answer: {final_answer}")
                break

            action = planned["action"]
            action_input = planned["action_input"]
            thought = planned.get("thought", "")

            observation = self._execute_tool(action, action_input)

            self.history.append(AgentStep(
                step=step_idx,
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation,
            ))

            if self.verbose:
                print(f"\nStep {step_idx} | Action: {action}({action_input})")
                print(f"Observation: {observation}")

        if final_answer is None:
            final_answer = "模拟序列未包含 final_answer。"

        return {
            "final_answer": final_answer,
            "steps": self.history,
            "total_steps": len(self.history),
        }


# ---------------------------------------------------------------------------
# 模块自检
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Agent 模块初始化完成")
    print(f"默认工具数: {len(DEFAULT_TOOLS)}")
    for tool in DEFAULT_TOOLS:
        print(f"  - {tool.name}: {tool.description[:60]}...")

    print("\n--- 工具自检 ---")
    print(f"calculator('2+3*4') = {_calculator('2+3*4')}")
    print(f"search_knowledge('监督学习') = {_search_knowledge('监督学习')}")
    print(f"get_current_time() = {_get_current_time()}")

    print("\n--- 模拟 Agent Loop（不加载模型）---")
    agent = Agent(verbose=False)
    agent.register_tools(DEFAULT_TOOLS)
    result = agent.simulate(
        task="计算 15*23 并查监督学习的定义",
        tool_sequence=[
            {"action": "calculator", "action_input": '{"expression": "15*23"}',
             "thought": "需要先计算 15*23。"},
            {"action": "search_knowledge", "action_input": '{"query": "监督学习"}',
             "thought": "计算完成，接下来查监督学习。"},
            {"final_answer": "15*23=345。监督学习是使用带标签数据训练模型的方法，"
                             "常见任务包括分类和回归。",
             "thought": "信息齐全，可以给出最终答案。"},
        ],
    )
    print(f"\n最终答案: {result['final_answer']}")
    print(f"总步数: {result['total_steps']}")
