# 02: Loop Termination — When Does the Agent Stop?

> **Part of the [Loop Engineering](./00-readme.md) notes.** A good loop must *always be able to stop*. This note is about every stopping condition you control.

---

## The Two Kinds of Stop

1. **The happy stop** — the model decided it has enough to answer. The loop exits because the model emitted a **final answer** (no tool call, or an explicit `END`).
2. **The forced stop** — the loop hit a **bound** enforced by code: max steps, max tokens, a timeout, or an interrupt. These keep the loop from running forever.

Both are engineering. Relying *only* on the model to stop is not engineering — it is hope.

## Normal Termination: Reading the Model's Intent

LangGraph and `create_agent` end when the model returns a message that is *not* a tool call (a plain assistant turn). Some models can also return an explicit **stop** / **quit**. Two things to verify on every run:

```python
# The two signals that end a normal loop
if output.content and not output.tool_calls:   # final answer, done
    return output
```

Guard the "I am done" case too, because agents like to announce "done" *and* call more tools:

```python
if output.tool_calls:      # never end while tools remain pending
    keep_looping = True
```

---

## Forced Stop #1: Maximum Steps

Every framework has a step cap. In LangGraph you add it to the graph or use a constant:

```python
from langgraph.graph import END, StateGraph

MAX_STEPS = 15

def should_continue(state):
    if state["step"] >= MAX_STEPS:
        return "force_stop"
    return "loop" if state["messages"][-1].tool_calls else END

builder.add_conditional_edges("agent", should_continue,
                              {"force_stop": "force_stop_node",
                               "loop": "tools", "agent": END})
```

When the cap trips, *say so* in the final message. A user should never see a silent truncation as if it were a complete answer:

```python
def force_stop_node(state):
    return {"messages": [{
        "role": "assistant",
        "content": "I stopped after {} steps to keep this bounded. "
                   "Here is my best partial answer: {}".format(MAX_STEPS, partial)}
    ]}
```

---

## Forced Stop #2: Token and Time Budgets

Even a small number of tool rounds can burn a huge number of tokens. Bound the *spend*, not just the *steps*:

- **Token cap per run** — check cumulative usage every iteration.
- **Wall-clock timeout** — a run that is "thinking" too long should be cut.
- **Per-tool budget** — e.g., "costly tools may be called at most N times".

```python
from datetime import datetime, timedelta

class Budget:
    def __init__(self, max_steps=20, max_tokens=80_000, max_seconds=120):
        self.deadline = datetime.utcnow() + timedelta(seconds=max_seconds)

    def exceeded(self, steps, tokens):
        if steps >= self.max_steps: return "steps"
        if tokens >= self.max_tokens: return "tokens"
        if datetime.utcnow() > self.deadline: return "time"
        return None
```

**Apply the budget in middleware or in the loop node** so every model invocation checks it first (see [04 - Loop Boundaries](./04-loop-boundaries.md)).

---

## Forced Stop #3: Human Interrupt (HITL)

The most user-friendly stop is a human. LangGraph's `interrupt()` pauses the loop and *resumes* it later with the human's decision. This is both a stop condition and a steering gate — see ch 18 and the projects.

```python
from langgraph.types import interrupt

def hitl_gate(state):
    if state["requires_approval"]:
        decision = interrupt({"msg": "Approve changing setpoint to 4200 rpm?",
                              "value": state["proposal"]})
        return {"approved": decision == "approve"}
    return state
```

HITL is stop-and-resume; a timeout/step stop is stop-and-recover. Decide which you want per danger level (destructive tools → HITL; math → no HITL).

---

## The Complete Stop Checklist

| Condition | Enforced by | What happens |
|-----------|-------------|--------------|
| Model gives final answer | model output | normal end |
| Step cap reached | graph / loop | forced stop, partial answer |
| Token cap reached | middleware | forced stop |
| Wall-clock timeout | middleware | forced stop |
| Too many failures | recovery (ch 03) | give up gracefully |
| Human approves/rejects | interrupt() | resume or terminate |

---

## Common Mistakes

- Trusting the model to always produce final answers (it won't).
- A token/step cap with **no path to a graceful partial answer** — the user gets nothing.
- Counting steps but not tokens (a single long tool result can blow the budget).
- Ending on a model that *still has* pending tool calls.

---

## What You Learned

- Normal stop = model final answer; forced stop = code bound.
- How to impose step, token, time, and HITL stops.
- The budget pattern that checks every iteration.
- Gracefully reporting a forced stop to the user.

**Next:** [03 - Error Recovery](./03-error-recovery.md) — what the loop does when an action fails.