# 03: Tools — Exposing Functions to the Model

> **Part of the [Harness Engineering](./00-readme.md) notes.** Tools are how the harness gives the model the ability to *act*. Build on ch 05, 06, and 22 (MCP).

---

## Converting a Function into a Tool

LangChain turns a Python function into a tool by reading its **docstring and type hints**: they become the JSON schema the model sees.

```python
from langchain.tools import tool

@tool
def fetch_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: Name of the city.
    """
    return weather_api.get(city)
```

The docstring is the tool's **instructions to the model**. Write it as: *what it does, what args mean, and when to use it.* The type hints define the schema.

---

## What Makes a Tool Good (Schemas Matter)

The model can only call what it understands. A great tool description is:

- **Specific** on *when to use it* ("Use for lookup by city; not for forecasts").
- **Courteous** about args and constraints ("value must be 0-5000").
- **Honest** about failure (returns an error string, doesn't raise silently).

```python
@tool
def write_setpoint(tag: str, value: float) -> str:
    """Write a PLC setpoint; only within the safe range.

    Args:
        tag: PLC tag, e.g. 'motor.speed'.
        value: Target value. MUST be within [0, 5000]; out-of-range is refused.
    """
    if not 0 <= value <= 5000:
        return f"REFUSED: {value} out of range [0, 5000]"
    plc.write(tag, value)
    return f"OK: wrote {value} to {tag}"
```

Compare to `def do(x): ...` — useless. Good prose beats terse signatures.

## The Tool Schema

Each tool exposes a schema the model reasons over. You can inspect and customize it:

```python
print(fetch_weather.params)      # see the generated JSON schema
```

Ch 19 (structured output) and ch 22 (MCP) both rest on this: the model must *decide* from the schema, so keep it minimal and clear. Too many params → confused calls.

## Tools and the Harness

- **Middleware guards tools** (note 02): permission, rate, size.
- **Tools feed the loop** (sibling [loop](../loop-engineering)): their results are re-read next round, so return clean, bounded, non-adversarial text (ch 32).
- **MCP tools** (ch 22) attach external servers the same way — `mcp_tools` from an `FastMCP` server.

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

mcp_client = MultiServerMCPClient({"plc": {"url": "http://localhost:9000/sse"}})
mcp_tools = await mcp_client.get_tools()          # same shape as local tools
all_tools = [fetch_weather, write_setpoint, *mcp_tools]
```

## Common Mistakes

- Vague or missing docstrings → the model guesses and misuses tools.
- Throwing exceptions from tools → the loop can't recover gracefully (return error strings).
- Returning massive/raw output → it re-enters context every round, inflating tokens.
- A tool named generically ("run", "do") with no guidance.

---

## What You Learned

- A function + docstring + type hints becomes a tool.
- Tool descriptions steer the model; write them well.
- Inspect/customize the schema; attach MCP tools the same way.
- Close the loop: keep tool output clean and bounded.

**Next:** [04 - Memory](./04-memory.md) — holding state across calls and turns.