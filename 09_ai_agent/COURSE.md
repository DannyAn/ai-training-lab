# 09 AI Agent 讲师脚本

建议时长：90 分钟。课前阅读：[README.md](./README.md)。

## 本节目标

学员能解释 Agent Loop 的核心循环（Thought → Action → Observation → Final Answer），区分 Tool、Function Calling 和 Skill 的概念，并理解工具描述如何影响模型行为。

## 开场问题

提问："现在你有一个 LLM、一个计算器函数、一个知识库搜索函数。用户问'计算 15*23 并告诉我监督学习的定义'。模型如何知道先调计算器还是先查知识库？如果计算器返回了结果，模型怎么知道下一步该查知识库而不是直接回答？"

引出 Agent Loop：模型不是一次性推理，而是在思考和工具调用之间循环。

## 课程定位：RAG 到 Agent 的桥梁

```text
08 RAG：给模型"外部记忆"      （检索知识库 → 回答问题）
09 Agent：给模型"手和决策权"    （自主选择工具 → 逐步完成复杂任务）
```

RAG 回答前一定会检索，Agent 自主决定**要不要检索、检索什么、检索后还要不要算数**。

用一个例子说明区别：
- 08 RAG 收到"监督学习是什么"→ 不管有没有用，先检索 → 把文档拼进 prompt → 生成答案。
- 09 Agent 收到"昨天股价涨了 5%，今天又涨了 3%，总共涨了多少？"→ 思考：需要查股价 → 调 search 工具 → 拿到数据 → 思考：需要算百分比 → 调 calculator → 得到结果 → 给出答案。

## 板书

用三种颜色区分：

```text
蓝色(模型推理)：
  Thought: 需要先算 15*23 → Action: calculator → 拿到 345
  Thought: 还需要监督学习定义 → Action: search_knowledge → 拿到定义
  Thought: 信息齐全 → Final Answer

紫色(工具执行)：
  calculator("15*23") → "345"
  search_knowledge("监督学习") → "监督学习使用带标签..."

绿色(循环控制)：
  Step 1 → Step 2 → Step 3 → 终止
  max_steps=8（安全阀）
```

标注三个核心组件：
1. **Agent Loop**：for 循环控制迭代，每步追加历史到 prompt。
2. **Function Calling**：正则解析 Thought/Action/Action Input → 执行 Python 函数 → 返回 Observation。
3. **Tool**：name + description（进入 prompt）+ parameters + func（实际执行）。

## 代码走读

### model.py

1. `Tool` dataclass（第 26-47 行）：四个字段各司其职。name 是模型调用时的标识，description 是 prompt 的组成部分，parameters 告诉模型传什么参数，func 是实际执行的 Python 函数。
2. `_calculator`（第 55-67 行）：安全执行数学表达式。用受限 `eval`，不是 `exec`。允许 math 函数，禁止 builtins。
3. `_search_knowledge`（第 70-86 行）：模拟知识库。字典硬编码了 6 个 ML 概念。生产环境应替换为 RAG 检索。
4. `AGENT_SYSTEM_PROMPT`（第 101-118 行）：定义 Agent 的行为规范（如何输出、何时终止）。工具描述通过 `{tool_descriptions}` 注入。
5. `Agent.__init__`（第 195-224 行）：延迟加载模型，先注册工具。`max_steps=8` 是安全阀。
6. `_build_prompt`（第 237-257 行）：每一步把所有历史（Thought + Action + Observation）追加到 messages，再交给 tokenizer apply_chat_template。
7. `_parse_response`（第 261-289 行）：正则解析两种模式（Final Answer / Thought + Action + Action Input）。这是 Function Calling 的文本解析实现。
8. `run()` （第 305-366 行）：主循环。构建 prompt → 推理 → 解析 → 执行工具 → 追加历史 → 循环或终止。
9. `simulate()` （第 376-420 行）：不加载模型，按预设工具序列走一遍循环。教学价值：先理解机制，再接入真实 LLM。

### train.py

四个演示函数，按复杂度递增：

1. `demo_simulate()`：模拟 Agent Loop，不加载模型。展示 3 步（calculator → search_knowledge → Final Answer）。
2. `demo_tool_execution()`：独立测试每个工具的输入输出。Agent 开发铁律——先调试工具。
3. `demo_custom_tool()`：演示如何注册自定义工具（只需写函数 + Tool 描述 + register）。
4. `demo_live_agent()`：加载 Qwen2.5-1.5B-Instruct 运行真实 Agent。--live 参数触发。

## 现场演示

```bash
# 1. 先看模块定义和工具自检
python 09_ai_agent/model.py

# 2. 模拟 Agent Loop（不加载模型，秒级）
python 09_ai_agent/train.py --sim

# 3. 完整演示（默认模式：模拟 + 工具演示）
python 09_ai_agent/train.py

# 4. 如果有 GPU/足够 RAM，加载真实 LLM
python 09_ai_agent/train.py --live
```

演示顺序必须是：

1. 运行 `model.py`，观察 Tool 的结构和工具自检结果。
2. 运行 `--sim`，手动标注每一步属于哪个阶段（Thought / Action / Observation）。
3. 强调：这 3 步的决策是人预设的（`tool_sequence`），真实 Agent 中这些决策由 LLM 做出。
4. 运行完整默认模式，观察三种演示的递进关系。
5. 最后才加载 LLM（如果有资源），展示同一任务模型可能选择不同的工具调用顺序。

## 必讲误区

- Agent 不是新的模型架构。它是在现有 LLM 外面包了一层**决策循环 + 工具执行**。
- Function Calling 不是模型"调用"了函数。是模型输出文本 → 解析器提取函数名和参数 → 系统执行 Python 函数。
- 工具描述不是给程序员看的注释，是给模型看的 prompt。描述模糊 → 模型选错工具。
- `simulate()` 不能验证模型的工具选择能力。它验证的是循环机制和工具执行逻辑。
- `max_steps` 是必须的安全阀。没有它，小模型可能无限循环调用同一工具。
- 当前实现使用文本正则解析，不是 structured output / JSON mode。小模型输出格式不稳定是预期现象。
- 工具执行在模型"外部"。模型看不到工具的内部逻辑，只能看到返回的字符串（Observation）。
- Agent 的能力由工具决定，不是由模型大小单一决定。给 Qwen2.5-1.5B 配备好的工具描述，可能比给大模型配备模糊的工具描述更有效。

## 课堂实验

### 实验 1：新增工具

学员新增一个 `word_count(text)` 工具：
1. 写一个 `_word_count(text: str) -> str` 函数。
2. 创建 `Tool(name="word_count", description=..., parameters=..., func=_word_count)`。
3. 注册到 Agent 并在模拟器中使用。

### 实验 2：工具描述对比

同时注册两个描述不同但功能相同的搜索工具：
- `search_knowledge`：精确描述（如"在知识库中搜索机器学习相关概念的定义"）。
- `wiki_search`：模糊描述（如"搜索信息"）。

观察模拟器中两者的区别，讨论：如果换成真实 LLM，模型会偏好哪一个？

### 实验 3：错误恢复

让 Agent 调用一个必定失败的工具（如除零或文件不存在），观察 Observation 中的错误信息。讨论：模型如何从错误中恢复？当前实现是否正确传递了错误上下文？

## 提问

1. Agent Loop 中每一步的 Thought/Action/Observation 是如何进入下一轮 prompt 的？
2. 为什么工具描述是 prompt engineering 的一部分？
3. `max_steps` 设为 1 和设为 20，分别有什么问题？
4. `simulate()` 和 `run()` 的关键区别是什么？各自测试什么？
5. 如果模型编造了一个不存在的工具名，系统会怎样响应？
6. 08 RAG 的检索和 09 Agent 的工具调用，在"谁决定调用"上有什么本质区别？

## 收束

用能力层次总结课程的演进：

```text
01-04 训练单一任务模型            → 处理固定类型输入输出
05    从零训练语言模型            → 自回归生成
06    预训练模型微调              → 迁移通用知识到下游任务
07    LoRA 参数高效微调           → 低成本改变模型行为
08    RAG 外部知识检索           → 不训练就能访问私有知识
09    Agent 自主决策循环          → 模型决定何时用什么工具
```

Agent 不是终点，而是新的起点：模型、工具、记忆、规划、多 Agent 协作的工程组合。

建议课后延伸阅读（不要求在课上讲完）：
- ReAct 论文（Yao et al., 2023）：Synergizing Reasoning and Acting in Language Models
- Toolformer / Gorilla / function-calling 的技术报告
- LangChain / LlamaIndex / CrewAI 的 Agent 模块文档
