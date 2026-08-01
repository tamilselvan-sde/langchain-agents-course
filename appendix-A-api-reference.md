# Quick API Reference

> **Goal:** One-stop lookup for every parameter, decorator, accessor, and type you will use across the course. Bookmark this page.
> **Previous chapter:** [Project 5 - PLC Diagnostic Agent](./project-5-plc-diagnostic-agent.md)
> **Next chapter:** [Appendix B - Troubleshooting Guide](./appendix-B-troubleshooting.md)

---

## How To Read This Reference

Everywhere in the course we use the same model setup:

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(
    model="openai/gpt-oss-120b",
    model_provider="openai",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0.3,
)
```

This appendix lists what you can pass to `create_agent`, `@tool`, what you can read from `ToolRuntime`, and what each middleware, message, and return type actually does.

---

## 1. `create_agent()` Parameters

```python
from langchain.agents import create_agent

agent = create_agent(
    model=...,                # required
    tools=...,                # required
    system_prompt=...,
    middleware=...,
    response_format=...,
    checkpointer=...,
    context_schema=...,
    name=...,
    store=...,
    state_schema=...,
)
```

| Parameter | Type | Required | Default | Purpose |
|---|---|---|---|---|
| `model` | `BaseChatModel` | ✅ Yes | — | The LLM the agent calls. Usually `init_chat_model(...)` for Groq `openai/gpt-oss-120b`. |
| `tools` | `list[Tool or BaseTool]` | ✅ Yes | `[]` | Tools the agent may call. Pass `[]` for a chat-only agent. |
| `system_prompt` | `str` or `Callable` | ❌ No | `None` | Sets agent behaviour. A string becomes the system message. A callable gets `(state)` and returns a string — useful for dynamic prompts. |
| `middleware` | `list[Middleware or Callable]` | ❌ No | `[]` | Runs before/after each model call and tool call. Order matters: outermost first. |
| `response_format` | `type[BaseModel]` or `dict` or `Literal["text", "json"]` | ❌ No | `None` | Forces structured final output. Pass a Pydantic class to get typed JSON. |
| `checkpointer` | `BaseCheckpointSaver` | ❌ No | `None` | Saves short-term memory per `thread_id`. `None` = stateless. Use `InMemorySaver()` for dev, `SqliteSaver`/`PostgresSaver` for prod. |
| `context_schema` | `type[BaseModel]` or `TypedDict` | ❌ No | `None` | Defines the trusted context object passed into `ToolRuntime.context`. Never filled by the model. |
| `name` | `str` | ❌ No | `None` | Agent name. Surfaces in LangSmith traces and multi-agent routing. |
| `store` | `BaseStore` | ❌ No | `None` | Long-term key-value memory shared across threads. Use `InMemoryStore()` for dev. |
| `state_schema` | `type` or `TypedDict` | ❌ No | `AgentState` | Adds custom fields to the agent's state (e.g. `notes`, `pending_approvals`). |

### Example with Every Parameter Set

```python
from typing import TypedDict
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_community.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

class AppState(TypedDict):
    notes: str
    pending_approvals: list[str]

class RequestContext(BaseModel):
    user_id: str
    environment: str
    db_url: str

class FinalReport(BaseModel):
    summary: str
    severity: str
    action_items: list[str]

@tool
def echo(text: str) -> str:
    """Echo back text."""
    return text

agent = create_agent(
    model=init_chat_model(
        model="openai/gpt-oss-120b",
        model_provider="openai",
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0.3,
    ),
    tools=[echo],
    system_prompt="You are a careful assistant.",
    middleware=[],
    response_format=FinalReport,
    checkpointer=InMemorySaver(),
    context_schema=RequestContext,
    name="echo_agent",
    store=InMemoryStore(),
    state_schema=AppState,
)
```

---

## 2. `@tool` Decorator Options

```python
from langchain_core.tools import tool

@tool(
    name="custom_name",
    description="What this tool does, shown to the model.",
    args_schema=MyArgs,         # optional Pydantic class for strict validation
    return_direct=True,         # optional: tool result becomes final answer
)
def my_tool(query: str, limit: int = 10) -> str:
    """Short description. Long-form docs go in the docstring."""
    return "..."
```

| Option | Type | Required | Default | Purpose |
|---|---|---|---|---|
| `name` | `str` | ❌ No | function name | Name the model sees in tool listings. Must be unique per agent. |
| `description` | `str` | ❌ No | from docstring | Overrides the docstring. This is what helps the model pick the right tool. |
| `args_schema` | `type[BaseModel]` | ❌ No | auto-derived | Strict Pydantic schema. Adds validation, defaults, and richer descriptions per field. |
| `return_direct` | `bool` | ❌ No | `False` | If `True`, the tool's return value becomes the final answer. The model never re-summarises it. Use for "render this raw JSON" tools. |

### Rules of Thumb

- Every parameter **must** have a Python type hint or schema generation fails.
- The first line of the docstring becomes the tool description unless `description` overrides it.
- `return_direct=True` skips the final LLM call — faster, but the model cannot summarise the result.

---

## 3. `ToolRuntime` Accessors

`ToolRuntime` is the trusted context object available inside `@tool` functions:

```python
from langchain_core.tools import tool, ToolRuntime

@tool
def who_am_i() -> str:
    """Return caller info."""
    runtime = ToolRuntime()
    return runtime.context.user_id
```

| Accessor | Type | Set By | Used For |
|---|---|---|---|
| `runtime.state` | `AgentState` (dict-like) | LangGraph automatically per `thread_id` | Reading/writing custom state fields from inside a tool. |
| `runtime.context` | `context_schema` instance | Caller passes it when invoking the agent | Trusted data: user IDs, secrets, environment flags. Never filled by the model. |
| `runtime.store` | `BaseStore` | Configured via `create_agent(store=...)` | Long-term cross-thread memory (notes, profiles, learned skills). |
| `runtime.stream_writer` | `Callable[[dict], None]` | LangGraph automatically | Emit custom tokens to the stream mid-tool-run (progress bars, partial JSON). |
| `runtime.execution_info` | `dict` | LangGraph automatically | Metadata about the current run: step number, recursion limit, run ID. |
| `runtime.tool_call_id` | `str` | LangGraph automatically | Matches this tool call to a `ToolMessage`. Needed when you build custom multi-tool orchestration. |

### Common Pattern: Streaming From a Tool

```python
@tool
def slow_search(query: str) -> str:
    """Search progressively streams results."""
    runtime = ToolRuntime()
    for i, chunk in enumerate(slow_db_scan(query)):
        runtime.stream_writer({"progress": f"row {i}"})
    return "done"
```

---

## 4. Middleware Catalog

Middleware transforms requests and responses around model calls and tool calls.

| Middleware | Purpose | When to Use |
|---|---|---|
| `SummarizationMiddleware` | Compresses old messages when token count exceeds a threshold. | Long-running sessions with limited context windows. |
| `ModelRetryMiddleware` | Catches transient model errors (timeouts, 429s) and retries with backoff. | Any agent hitting a flaky or rate-limited Groq endpoint. |
| `ToolRetryMiddleware` | Re-runs a failing tool with the same arguments a fixed number of times. | Tools that call flaky APIs (network blips). |
| `ToolListMiddleware` | Dynamically changes which tools the agent sees per turn. | Large tool catalogues — only show relevant tools, hide the rest. |
| `TodoListMiddleware` | Maintains a live to-do list inside agent state. | Long multi-step tasks where the agent tends to lose the thread. |
| `SubAgentMiddleware` | Spawns a sub-agent for delegated sub-tasks and returns its result. | Multi-agent patterns with a parent "manager" agent. |
| `FilesystemMiddleware` | Gives the agent controlled read/write access to a sandboxed directory. | Coding agents, log analyzers, doc generators. |
| `MemoryMiddleware` | Auto-remembers facts about the user in the long-term `store`. | Personalised assistants, customer support agents. |
| `SkillsMiddleware` | Loads `.md` skill files from disk into the agent's prompt at runtime. | Bootstrap-style agents whose capabilities change without code edits. |
| `guardrails` (factory) | Wraps a producer function that validates/prompts back the user input. | Any public-facing agent that must reject unsafe input. |
| `HITL` (Human-in-the-Loop) | Pauses the agent at a configurable checkpoint and resumes after human approval. | Risky actions: deletions, payments, infrastructure changes. |

### Typical Composition

```python
from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
    ModelRetryMiddleware,
    ToolRetryMiddleware,
    MemoryMiddleware,
)
from langgraph.pregel import Guardrail  # or your custom guardrail factory

middleware = [
    Guardrail(producer=lambda x: x),       # outermost: validate input first
    ModelRetryMiddleware(max_retries=3),
    SummarizationMiddleware(max_tokens=4000),
    ToolRetryMiddleware(max_retries=2),
    MemoryMiddleware(),                    # innermost: memory is closest to model
]
```

---

## 5. Message Types

| Message | Source | Required Fields | Common Use |
|---|---|---|---|
| `HumanMessage` | User input. | `content` (str or list) | What the user said. |
| `AIMessage` | Model output. | `content` + optional `tool_calls` | Text the model produced, plus any tool calls it requested. |
| `SystemMessage` | System prompt. | `content` | Sets agent persona/rules. Usually injected by `system_prompt`. |
| `ToolMessage` | Tool result. | `content` + `tool_call_id` | The result of an executed tool, paired back to its original `tool_calls` entry by ID. |

```python
from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage, ToolMessage,
)

history = [
    SystemMessage(content="You are a terse assistant."),
    HumanMessage(content="What is 2+2?"),
    AIMessage(content="", tool_calls=[{"name": "calc", "args": {"x": 2, "y": 2}, "id": "c1"}]),
    ToolMessage(content="4", tool_call_id="c1"),
    AIMessage(content="The answer is 4."),
]
```

---

## 6. Tool Return Types

A `@tool` function may return any of the following. The return type controls how LangGraph turns the value into a state update.

| Return Type | Behaviour | Example |
|---|---|---|
| `str` | Becomes the `ToolMessage.content` directly. | `return "42"` |
| `dict` | Treated as a state patch; keys must match `AgentState` fields (or `update` keys). Special keys: `update`, `messages`, `context`. | `return {"messages": [...], "update": {"counter": 1}}` |
| `Command` | Explicit object with `update`, `goto`, `resume`, and `interrupt` fields. Most expressive form. | `return Command(update={...}, goto="some_node")` |
| `list[dict]` | A batch of state patches applied in order. | `return [{"update": {...}}, {"messages": [...]}]` |
| `return_direct=True` + any return | Tool result skips the final model summarisation and becomes the agent's final answer. | `@tool(return_direct=True) def get_raw(): ...` |

### `Command` Examples

```python
from langgraph.types import Command

@tool
def done_with_summary(summary: str) -> Command:
    """End the run and store the summary."""
    return Command(
        update={"messages": [{"role": "assistant", "content": summary}],
                "notes": summary},
        goto="END",
    )

@tool
def escalate(ticket: str) -> Command:
    """Hand control to the manager agent."""
    return Command(goto="manager_agent")
```

---

## 7. Quick Lookup Matrix

| You want to... | Use |
|---|---|
| Make the agent remember chat history across turns | `checkpointer=InMemorySaver()` and pass `config={"configurable": {"thread_id": "user-123"}}` |
| Make the agent remember facts across threads | `store=InMemoryStore()` + `MemoryMiddleware()` |
| Add private state fields visible to tools | `state_schema=MyState` |
| Add trusted caller context | `context_schema=MyContext` + pass `context=` at invoke time |
| Force JSON output | `response_format=MyPydanticModel` |
| Stream tokens to UI | `for chunk in agent.stream(...)` |
| Stream custom progress from a tool | `runtime.stream_writer({"label": "..."})` |
| Pause for human approval | `HITL` middleware + `interrupt()` in tools |
| Dynamic tool hint swapping | `ToolListMiddleware` |
| Retry on Groq 429 | `ModelRetryMiddleware(max_retries=3)` |
| Compress long chats | `SummarizationMiddleware(max_tokens=4000)` |
| Skip model re-summarisation | `@tool(return_direct=True)` |

---

> **Next:** [Appendix B - Troubleshooting Guide](./appendix-B-troubleshooting.md) covers the errors you will hit and how to fix them.