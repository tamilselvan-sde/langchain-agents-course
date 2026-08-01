# Troubleshooting Guide

> **Goal:** The most common errors from this course, why they happen, and the one-line fix.
> **Previous chapter:** [Appendix A - API Reference](./appendix-A-api-reference.md)
> **Next chapter:** [Appendix C - Production Directory Structure Guide](./appendix-C-directory-structure.md)

---

## How To Use This Guide

Errors are grouped by their root cause, not by where they表面. Look for the **error message pattern** first; the cause and fix follow. Always read the full traceback to the last line — the actual cause is the last `Exception:` line, not the first.

> Tip: Turn on LangSmith tracing (`LANGSMITH_TRACING=true`) to see exactly which step raised. Most ambiguous errors become obvious once you see the failing node.

---

## 1. Groq-Specific Issues

### 1.1 Model Not Found

**Error message:**
```
openai.NotFoundError: Error code: 404 - {'error': {'message': 'The model openai/gpt-oss-120b does not exist'}}
```

**Cause:** The `model` string is wrong, or your Groq preview access has not been enabled. Groq preview model names contain a provider prefix (`openai/`) — the raw OpenAI name `gpt-oss-120b` will not match.

**Fix:**
- Use the exact string `model="openai/gpt-oss-120b"`.
- Make sure you set `model_provider="openai"` and `base_url="https://api.groq.com/openai/v1`.
- Confirm your Groq account has preview-model access at `https://console.groq.com`.

```python
model = init_chat_model(
    model="openai/gpt-oss-120b",
    model_provider="openai",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
)
```

---

### 1.2 Authentication Errors

**Error message:**
```
openai.AuthenticationError: Error code: 401 - {'error': {'message': 'Invalid API Key'}}
```

**Cause:** `GROQ_API_KEY` is missing, expired, or was copied with a leading space. The variable is also sometimes set only in your IDE and not in the shell that runs the script.

**Fix:**
```bash
# Confirm the key is exported in the SAME shell that runs python
export GROQ_API_KEY="gsk_..."           # no leading/trailing spaces
echo $GROQ_API_KEY | wc -c              # should be ~57 chars
python -c "import os; print(len(os.environ['GROQ_API_KEY']))"
```
If you use `python-dotenv`, make sure the `.env` file is loaded **before** `init_chat_model` runs.

---

### 1.3 Rate Limits (429)

**Error message:**
```
openai.RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached...'}}
```

**Cause:** Free Groq tier limits requests per minute and tokens per minute. Burst agent loops (especially multi-tool turns) trigger these limits.

**Fix:**
- Add `ModelRetryMiddleware(max_retries=5, initial_delay=2.0)` — it backs off exponentially.
- Lower `temperature` to reduce token jitter.
- Cache responses with `langchain_community.cache.InMemoryCache`.
- Run fewer agents in parallel.

```python
from langchain.agents.middleware import ModelRetryMiddleware

agent = create_agent(
    model=model,
    tools=[...],
    middleware=[ModelRetryMiddleware(max_retries=5, initial_delay=2.0)],
)
```

---

### 1.4 Context Window Exceeded

**Error message:**
```
openai.BadRequestError: Error code: 400 - {'error': {'message': 'prompt is too long: 130000 tokens > 120000 maximum'}}
```

**Cause:** Conversation history outgrew the model context (`gpt-oss-120b` has a hard cap).

**Fix:**
```python
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model=model,
    tools=[...],
    middleware=[SummarizationMiddleware(max_tokens=60000)],
)
```
Set `max_tokens` to roughly half the model window so summarisation triggers early.

---

## 2. LangChain Import Errors

### 2.1 Wrong Module

**Error message:**
```
ImportError: cannot import name 'create_agent' from 'langchain.agents'
```

**Cause:** You are on LangChain `<1.0`. Pre-1.0 used `initialize_agent` (now removed).

**Fix:**
```bash
pip install -U "langchain>=1.0" "langgraph>=0.2"
```
```python
# OLD (removed):
from langchain.agents import initialize_agent  # ❌

# NEW:
from langchain.agents import create_agent       # ✅
```

---

### 2.2 Deprecated Imports

**Error messages you will see in 1.x:**
```
ImportWarning: langchain.agents.agent_toolkits is deprecated; use langchain_community.tools instead.
```

**Cause:** Tool kits moved. Code from 2024 tutorials still imports the old paths.

**Fix:**
```python
# OLD:
from langchain.agents.agent_toolkits import FileToolkit  # ❌
from langchain.tools import BaseTool                       # ❌

# NEW:
from langchain_community.tools import tool                 # ✅
from langchain_core.tools import BaseTool                 # ✅
```

---

### 2.3 Circular Import on `ToolRuntime`

**Error message:**
```
ImportError: partially initialized module 'langchain_core.tools' has no attribute 'ToolRuntime'
```

**Cause:** You imported `ToolRuntime` at module top before LangChain finished wiring its lazy imports — common when your `tools.py` is imported by both `agent.py` and `tools_test.py`.

**Fix:** Move the import inside the function, or restructure so `tools.py` imports only from `langchain_core.tools`:

```python
# tools.py
from langchain_core.tools import tool, ToolRuntime   # at top is fine

@tool
def my_tool() -> str:
    """..."""
    runtime = ToolRuntime()       # accessed at call-time, import-time ordering is fine
    return runtime.context.user_id
```

---

## 3. Tool Schema Errors

### 3.1 Missing Type Hints

**Error message:**
```
ValueError: Tool function 'calculate' has no type annotations. Tools require type hints to build their schema.
```

**Cause:** The `@tool` decorator reads Python type hints to generate the JSON schema. A function with no annotations has nothing to introspect.

**Fix:** Add hints to **every** parameter and the return type:

```python
# BAD:
@tool
def calculate(x, y):                     # ❌
    return x + y

# GOOD:
@tool
def calculate(x: int, y: int) -> int:    # ✅
    """Add two integers."""
    return x + y
```

---

### 3.2 Reserved Parameter Names

**Error message:**
```
ValueError: Tool 'run' uses reserved parameter name 'state' without declaring it for runtime access.
```

**Cause:** Names like `state`, `context`, `store`, `runtime`, `config` are reserved by `ToolRuntime`. If your tool genuinely needs them, you must use `ToolRuntime` to access them — you cannot receive them as plain function arguments.

**Fix:**
```python
# BAD:
@tool
def get_session(state: dict) -> str:    # ❌ 'state' collides with runtime state
    return state["user_id"]

# GOOD:
from langchain_core.tools import tool, ToolRuntime

@tool
def get_session() -> str:               # ✅
    """Return the current user id."""
    return ToolRuntime().state.get("user_id", "anonymous")
```

---

### 3.3 Pydantic `args_schema` Mismatch

**Error message:**
```
pydantic.ValidationError: 1 validation error for MyArgs
query
  field required (type=value_error.missing)
```

**Cause:** The Pydantic schema you passed as `args_schema` requires a field that the function signature does not expose, or the model sent `null` for a non-optional field.

**Fix:** Make every required Pydantic field match a function parameter, and make optional fields `Optional[...]` with defaults:

```python
from typing import Optional
from pydantic import BaseModel, Field

class SearchArgs(BaseModel):
    query: str = Field(description="The text to search for.")
    limit: Optional[int] = Field(default=10, description="Max results.")

@tool(args_schema=SearchArgs)
def search(query: str, limit: int = 10) -> str:
    """Search documents."""
    return "..."
```

---

## 4. Memory Issues

### 4.1 `thread_id` Mismatch

**Symptom:** Agent acts amnesic — it forgets everything between turns even with `InMemorySaver()`.

**Cause:** You passed a different `thread_id` per call (or forgot to pass `config` at all).

**Fix:** Use the **same** `thread_id` for every turn of the same conversation:
```python
config = {"configurable": {"thread_id": "user-42-session-1"}}

agent.invoke({"messages": [HumanMessage("hi")]}, config=config)
agent.invoke({"messages": [HumanMessage("what did I just say?")]}, config=config)  # ✅ same id
```

---

### 4.2 Checkpointer Not Set

**Error message:**
```
ValueError: No checkpointer found. Set checkpointer=... when creating the agent to enable short-term memory.
```

**Cause:** You tried `agent.stream(..., interrupt_before=...)` or HITL without a checkpointer. State has nowhere to persist on pause.

**Fix:**
```python
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model=model,
    tools=[...],
    checkpointer=InMemorySaver(),   # ✅ required for HITL and resume
)
```

---

### 4.3 Store Reset Between Restarts

**Symptom:** Long-term `MemoryMiddleware` facts disappear after you restart the script.

**Cause:** `InMemoryStore()` lives only in process memory. It dies with the process.

**Fix:** Use a persistent store in any non-demo code:
```python
from langgraph.store.postgres import PostgresStore
# or
from langgraph.store.sqlite import SqliteStore

store = SqliteStore.from_conn_string("agent_memory.db")
```

---

## 5. MCP Connection Issues

### 5.1 Server Not Found

**Error message:**
```
ConnectionError: Could not connect to MCP server 'filesystem' at '/abs/path'
```

**Cause:** The `command`/`args` path does not resolve, or the server package is not installed.

**Fix:**
```bash
# Validate the path exists
ls /abs/path    # must exist
npx -y @modelcontextprotocol/server-filesystem --help   # must not error
```
```python
# Use absolute paths, not relative:
from langchain.agents.mcp import load_mcp_tools

tools = load_mcp_tools(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/Users/me/data"],
)
```

---

### 5.2 `npx` Not Installed

**Error message:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'npx'
```

**Cause:** Node.js / npm is not on your PATH (or not installed).

**Fix:**
```bash
# macOS
brew install node
# Ubuntu
sudo apt install -y nodejs npm
# Verify
which npx && npx --version
```

---

### 5.3 stdio Timeout

**Error message:**
```
TimeoutError: MCP server did not respond within 15s
```

**Cause:** The MCP server is slow to boot, often because `npx` is downloading the package on first run. The default 15s timeout is too short for cold starts.

**Fix:**
- Pre-install: `npm install -g @modelcontextprotocol/server-filesystem`.
- Bump the timeout:
```python
from langchain.agents.mcp import load_mcp_tools

tools = load_mcp_tools(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/data"],
    transport="stdio",
    timeout=60.0,           # ✅ seconds
)
```

---

## 6. Streaming Issues

### 6.1 Version Mismatch

**Error message:**
```
TypeError: agent.stream() got unexpected keyword argument 'stream_mode'
```

**Cause:** You are on `langgraph < 0.2`, where `stream_mode` was called `stream_options`.

**Fix:**
```bash
pip install -U "langgraph>=0.2"
```

---

### 6.2 Empty Chunks

**Symptom:** `for chunk in agent.stream(...): print(chunk)` prints `{}`.

**Cause:** Default `stream_mode` is `"values"` — it only emits when state changes. Tools that don't return anything produce empty diffs.

**Fix:** Pass multiple modes to see everything:
```python
for chunk in agent.stream(
    {"messages": [HumanMessage("hi")]},
    config,
    stream_mode=["messages", "updates", "values"],
):
    print(chunk)
```

---

### 6.3 Tool Output Not Streamed

**Symptom:** Model tokens stream fine, but tool output appears only at the very end.

**Cause:** Tool output is emitted on the `updates` channel, not `messages`.

**Fix:** Subscribe to both:
```python
for chunk in agent.stream(..., stream_mode=["messages", "updates"]):
    kind, payload = chunk
    if kind == "messages":
        print("model:", payload[0].content, end="")
    elif kind == "updates":
        print("tool:", payload)
```

---

## 7. Catch-All Debugging Recipe

When none of the above matches, run this checklist in order:

1. `python -c "import langchain, langgraph; print(langchain.__version__, langgraph.__version__)"` — both must be at least 1.0 / 0.2.
2. `echo $GROQ_API_KEY` — must print a 56-byte key starting with `gsk_`.
3. Run a one-line model check:
   ```python
   model.invoke("hello")
   ```
   If this fails the rest will too.
4. Set `LANGSMITH_TRACING=true` and re-run. Open the trace; the failing node is highlighted.
5. Strip your agent to **model + one tool + no middleware**. Rebuild one piece at a time until the error reappears — that piece is the cause.

---

> **Next:** [Appendix C - Production Directory Structure Guide](./appendix-C-directory-structure.md) walks through how to lay out a real LangChain project on disk.