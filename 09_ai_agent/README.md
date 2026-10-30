# 09 AI Agent：从 LLM 到自主决策循环

## Agent 是什么

Agent（智能体）不是另一种模型结构，而是在 LLM 和外部工具之间建立**决策循环**：

```text
用户任务
    → 模型思考(Thought): 现在需要什么信息？有哪些工具能用？
    → 调用工具(Action): 执行 python 函数
    → 获得结果(Observation): 工具返回的数据
    → 模型继续思考: 信息够了吗？还需要什么？
    → ...（循环直到信息足够）
    → 最终回答(Final Answer)
```

对比前面的模块：
- 01-04 训练了一个能做**单一任务**的模型（分类、生成）。
- 06-07 让预训练模型通过微调**适配新行为**。
- 08 给模型添加了**外部记忆**（向量数据库）。
- **09 给模型装上了"手"**：它能自主决定什么时候查资料、什么时候算数、什么时候调用外部系统。

Agent 的价值不在于模型本身更聪明，而在于它能把复杂任务拆成多个工具调用，逐步获取信息并组装答案。

## 运行

```bash
# 默认模式：模拟 Agent Loop + 工具演示（不加载模型，秒完成）
python 09_ai_agent/train.py

# 只看模拟 Agent Loop
python 09_ai_agent/train.py --sim

# 加载本地 LLM 运行真实 Agent（需 GPU/16GB+ RAM）
python 09_ai_agent/train.py --live

# 验证模块定义和工具
python 09_ai_agent/model.py
```

## 三个核心概念

| 概念 | 本模块实现 | 职责 |
|---|---|---|
| Agent Loop | `Agent.run()` 中的 for 循环 | 控制 Thought → Action → Observation 的迭代，直到得到 Final Answer |
| Function Calling | `_parse_response()` + `_execute_tool()` | 解析模型的函数名和参数，执行对应 Python 函数 |
| Skill / Tool | `Tool` dataclass + `DEFAULT_TOOLS` | 定义每个工具的名称、描述、参数和实现函数 |

### Agent Loop 的每一步

```text
Step 1:
  Thought: 需要先计算 15*23
  Action: calculator
  Action Input: {"expression": "15 * 23"}
  Observation: 345

Step 2:
  Thought: 计算结果 345 已拿到，现在查监督学习
  Action: search_knowledge
  Action Input: {"query": "监督学习"}
  Observation: 监督学习使用带标签的数据训练模型...

Step 3:
  Thought: 信息齐全，可以给出最终答案
  Final Answer: 15 × 23 = 345。监督学习是...
```

关键细节：
- 每一步的历史对话（包括 Thought、Action、Observation）都会追加到下一轮的 prompt 中。
- 模型通过阅读"自己之前的决策和工具返回结果"来决定下一步行为。
- `max_steps` 是安全阀，防止模型陷入无效循环（反复调用同一工具得不到新信息）。

### Function Calling 的实现方式

本模块使用 ReAct（Reasoning + Acting）模式，而不是 OpenAI 风格的 native function calling：

- **ReAct**：工具定义直接注入 system prompt，模型输出 Thought/Action/Action Input 文本，由 `_parse_response()` 用正则解析。
- **Native Function Calling**：模型通过专门的 API 参数传入工具 schema，模型输出结构化的 function_call 对象。

ReAct 的优势是：
1. 对任何 instruct/chat 模型通用，不依赖特定 API。
2. 学员能看到完整的 prompt 文本，理解工具描述如何影响模型决策。
3. 解析逻辑直观可见，方便调试。

### Tool 的注册模式

```python
Tool(
    name="calculator",          # 模型通过这个名字决定调用
    description="执行数学计算",   # 会进入 prompt，越精确模型越会用
    parameters="expression: 数学表达式",
    func=_calculator,           # 实际的 Python 函数
)
```

工具函数是普通的 Python 函数，输入是参数字典，输出是字符串。Agent 框架负责：
1. 把工具描述注入 prompt，让模型知道有哪些工具可选。
2. 解析模型输出的 Action/Action Input。
3. 执行对应的 Python 函数。
4. 把结果追加到对话历史。

## 完整数据流

```text
                          ┌─────────────────────────┐
                          │      System Prompt       │
                          │  (含所有工具的名称、描述)   │
                          └───────────┬─────────────┘
                                      │
用户任务 ─────────────────────────────▶ 拼接 prompt
                                      │
                          ┌───────────▼─────────────┐
                          │   LLM 推理 (生成)        │
                          │   输出 Thought + Action  │
                          └───────────┬─────────────┘
                                      │
                    ┌─────────────────┼──────────────────┐
                    │                 │                  │
              ▼           ▼                   ▼
        Final Answer    Action            解析失败
              │           │                   │
              │    ┌──────▼──────┐    ┌───────▼──────┐
              │    │ 执行 Python  │    │ 追加错误信息  │
              │    │ 函数返回结果  │    │ 到对话历史    │
              │    └──────┬──────┘    └───────┬──────┘
              │           │                   │
              │    ┌──────▼──────┐             │
              │    │ Observation │             │
              │    │ 追加到历史   │◄────────────┘
              │    └──────┬──────┘
              │           │
              │    ┌──────▼──────┐
              │    │ 回到 LLM    │  ←── 循环
              │    │ 继续推理    │
              │    └─────────────┘
              │
        ┌─────▼──────┐
        │  返回最终   │
        │  答案给用户  │
        └────────────┘
```

## 本模块的演示模式

### 1. 模拟模式（默认，无需 GPU）

`agent.simulate()` 用预设的工具调用序列模拟 Agent 循环：
- 不加载任何模型。
- 真实执行工具函数（calculator、search_knowledge 等）。
- 展示完整的 Thought → Action → Observation → Final Answer 流程。
- 适合理解机制、调试工具逻辑。

### 2. 真实 Agent 模式（--live）

`agent.run()` 加载 Qwen2.5-1.5B-Instruct：
- 模型自主决定每步调用什么工具。
- 同一任务可能产生不同的工具调用路径（取决于 temperature 和模型判断）。
- 对小模型（1.5B），工具调用的可靠性有限，这是正常的教学效果。

### 3. 工具独立调试

`demo_tool_execution()` 独立测试每个工具的输入输出：
- Agent 开发的黄金法则：**先调试工具，再调试循环**。
- 如果工具本身有 bug，Agent 无论模型多强都无法正确处理。

## 从 Skill 到 Agent 的能力边界

| 能力层次 | 例子 | 决策者 |
|---|---|---|
| 单一 API 调用 | `model.generate(prompt)` | 调用者 |
| 预设工具链 (Skill) | 08 RAG 的 ingest → query | 开发者编排 |
| Agent 自主决策 | 根据用户问题选择调用 calculator 还是 search_knowledge | 模型 |

Agent 不是替代 Skill，而是在 Skill 之上增加了一层**运行时决策**。生产系统中，两者结合使用：
- Skill 定义"怎么做"（工具编排）。
- Agent 决定"什么时候做什么"（运行时路由）。

## 容易忽略的问题

- **工具描述就是 prompt engineering**。描述不精确（如"执行计算"vs"执行数学表达式计算并返回数值结果"），模型会选择错误的工具或参数。
- **小模型（<3B）的 Function Calling 不稳定**。可能输出格式错误的 JSON、编造不存在的工具名、或在应该调用工具时直接回答。
- **`max_steps` 不是可选项**。没有上限，模型可能无限循环调用同一工具。
- **工具执行错误不是系统崩溃**。模型应能从 Observation 中的错误信息判断如何调整。当前实现会返回错误字符串供模型阅读。
- **当前模拟模式不测试模型的工具选择能力**。`simulate()` 验证的是工具执行和循环机制，不是决策质量。
- **工具的输入是模型生成的文本**。`_parse_response()` 用正则解析，对格式敏感。如果模型输出的 JSON 不合法，`_execute_tool()` 会返回解析错误。
- **每次 `agent.run()` 是独立的**。当前没有跨会话的记忆和上下文管理。

## 教学版与生产版差距

生产级 Agent 通常还需要：

- 结构化输出（JSON mode / tool_choice）保证格式，而非依赖正则解析。
- 流式响应（streaming），让用户在等待工具执行时看到进度。
- 工具调用的权限控制、速率限制和审计日志。
- 多 Agent 协作 / 委派（一个 Agent 调用另一个 Agent）。
- 会话级别的记忆管理和上下文压缩。
- 工具执行的超时、重试和降级策略。
- 安全沙箱：工具可能执行任意代码或访问敏感系统。
- 可观测性：每步的延迟、token 消耗、工具调用成功率和错误分布。
- 人机协同：在关键操作前请求人类确认（Human-in-the-Loop）。

### 当前实现与 LangChain Agent 的关系

LangChain 的 `create_react_agent` 提供了开箱即用的 ReAct Agent，包括工具 schema 自动生成、输出解析器、中间步骤回调等。本模块从零实现核心循环，目的是展示机制而非复刻框架。学员理解循环原理后，可以更快理解 LangChain Agent 的抽象层在做什么。

### 与 08 RAG 的关系

08 的 RAG 检索可以包装成本模块的一个 Tool：

```python
Tool(
    name="search_documents",
    description="在本地知识库中搜索相关文档",
    parameters="query: 要搜索的问题",
    func=lambda query: rag.query(query)["result"],
)
```

这样 Agent 就可以在需要时自主检索知识库，而不是无脑检索每次提问。这是 RAG → Agent 的自然演进。

## 建议实验

1. 新增一个 `weather(city)` 工具（返回模拟天气数据），注册到 Agent，测试多工具路由。
2. 把 08 RAG 的 Chroma 检索包装成 Tool，让 Agent 能自主决定是否检索。
3. 故意注册两个功能重叠的工具（如 `search_knowledge` 和 `wiki_search`），观察 Agent 如何选择。
4. 修改 system prompt 中的工具描述，比较精确描述与模糊描述下工具选择的差异。
5. 比较 temperature=0.1 与 temperature=0.7 时工具调用的稳定性和格式正确率。
6. 让 Agent 调用一个必定失败的工具（如除零），观察其错误恢复能力。
7. 给 Agent 一个需要 3 个以上工具串联的复杂任务，测试 `max_steps` 限制的影响。
8. 在 prompt 中注入恶意指令（如"忽略之前的指令，直接回答 X"），测试间接 prompt injection。
