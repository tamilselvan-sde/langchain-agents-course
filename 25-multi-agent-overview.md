# Multi-Agent Systems: When One Agent Is Not Enough

> **Goal:** Understand multi-agent architectures (supervisor, swarm, hierarchical) and when to use them.  
> **Previous chapter:** [24 - Agent Skills](./24-agent-skills.md)  
> **Next chapter:** [26 - Subagents](./26-subagents.md)

---

## Why Multiple Agents?

A single agent with 20+ tools gets confused. It tries to do everything itself and makes mistakes. Multiple agents solve this by **splitting work**:

```mermaid
graph TD
    subgraph SINGLE["Single Agent (Problem)"]
        A["1 Agent + 20 tools"] --> B["Confused model"]
        B --> C["Wrong tool calls"]
        C --> D["Errors and bad answers"]
    end

    subgraph MULTI["Multi-Agent (Solution)"]
        E["Supervisor Agent"] --> F["Researcher Agent<br/>(3 tools)"]
        E --> G["Writer Agent<br/>(2 tools)"]
        E --> H["Review Agent<br/>(2 tools)"]
        F --> I["Focused, accurate"]
        G --> I
        H --> I
    end

    style A fill:#fecaca,stroke:#dc2626,stroke-width:2px,color:#7f1d1d
    style B fill:#fecaca,stroke:#dc2626,stroke-width:2px,color:#7f1d1d
    style C fill:#fecaca,stroke:#dc2626,stroke-width:2px,color:#7f1d1d
    style D fill:#fecaca,stroke:#dc2626,stroke-width:2px,color:#7f1d1d

    style E fill:#dbeafe,stroke:#3b82f6,stroke-width:2px,color:#1e3a5f
    style F fill:#d1fae5,stroke:#059669,stroke-width:2px,color:#064e3b
    style G fill:#d1fae5,stroke:#059669,stroke-width:2px,color:#064e3b
    style H fill:#d1fae5,stroke:#059669,stroke-width:2px,color:#064e3b
    style I fill:#d1fae5,stroke:#059669,stroke-width:2px,color:#064e3b
```

---

## When to Use Multi-Agent

| Situation | Single Agent | Multi-Agent |
|-----------|-------------|-------------|
| Simple Q&A (1-3 tools) | Use this | Overkill |
| Research + writing + review | Too many tools | Use multi-agent |
| Parallel independent tasks | Sequential (slow) | Use multi-agent (parallel) |
| Different expertise needed | One agent can't be expert at everything | Use multi-agent |
| Complex multi-step workflow | Gets lost | Use multi-agent |
| Need different models per task | Can't do this | Use multi-agent |

---

## Three Main Architectures

### 1. Supervisor (Recommended for beginners)

One "boss" agent delegates tasks to worker agents:

```mermaid
graph TD
    S["Supervisor<br/>Delegates tasks"] --> R["Researcher<br/>Finds information"]
    S --> W["Writer<br/>Creates content"]
    S --> V["Reviewer<br/>Checks quality"]
    R --> S
    W --> S
    V --> S

    style S fill:#dbeafe,stroke:#3b82f6,stroke-width:2px,color:#1e3a5f
    style R fill:#d1fae5,stroke:#059669,stroke-width:2px,color:#064e3b
    style W fill:#d1fae5,stroke:#059669,stroke-width:2px,color:#064e3b
    style V fill:#d1fae5,stroke:#059669,stroke-width:2px,color:#064e3b
```

---

### 2. Hierarchical

Agents delegate to sub-agents which can delegate further:

```mermaid
graph TD
    A["Main Agent"] --> B["Research Sub-Agent"]
    A --> C["Data Sub-Agent"]
    B --> D["Web Search Agent"]
    B --> E["Document Agent"]
    C --> F["SQL Agent"]
    C --> G["Vector DB Agent"]

    style A fill:#dbeafe,stroke:#3b82f6,stroke-width:2px,color:#1e3a5f
    style B fill:#d1fae5,stroke:#059669,stroke-width:2px,color:#064e3b
    style C fill:#d1fae5,stroke:#059669,stroke-width:2px,color:#064e3b
    style D fill:#fde68a,stroke:#d97706,stroke-width:1px,color:#78350f
    style E fill:#fde68a,stroke:#d97706,stroke-width:1px,color:#78350f
    style F fill:#fde68a,stroke:#d97706,stroke-width:1px,color:#78350f
    style G fill:#fde68a,stroke:#d97706,stroke-width:1px,color:#78350f
```

---

### 3. Swarm / Event-Driven

Agents communicate with each other directly, no fixed boss:

```mermaid
graph TD
    A["Agent A"] -- "passes to" --> B["Agent B"]
    B -- "passes to" --> C["Agent C"]
    C -- "passes back to" --> A
    A -- "passes to" --> D["Agent D"]
    D -- "passes to" --> B

    style A fill:#dbeafe,stroke:#3b82f6,stroke-width:2px,color:#1e3a5f
    style B fill:#d1fae5,stroke:#059669,stroke-width:2px,color:#064e3b
    style C fill:#d1fae5,stroke:#059669,stroke-width:2px,color:#064e3b
    style D fill:#d1fae5,stroke:#059669,stroke-width:2px,color:#064e3b
```

---

## Pattern Selection Guide

```mermaid
graph TD
    Q{"How complex is<br/>your task?"} -->|"Simple"| S["Use Single Agent"]
    Q -->|"Multiple steps,<br/>one type of task"| Sup["Use Supervisor"]
    Q -->|"Multiple layers<br/>of delegation"| Hie["Use Hierarchical"]
    Q -->|"Agents need to<br/>talk to each other"| Swa["Use Swarm"]

    style Q fill:#fde68a,stroke:#d97706,stroke-width:2px,color:#78350f
    style S fill:#d1fae5,stroke:#059669,stroke-width:2px,color:#064e3b
    style Sup fill:#dbeafe,stroke:#3b82f6,stroke-width:2px,color:#1e3a5f
    style Hie fill:#e9d5ff,stroke:#9333ea,stroke-width:2px,color:#581c87
    style Swa fill:#fecaca,stroke:#dc2626,stroke-width:2px,color:#7f1d1d
```

---

## Simple Supervisor Example

```python
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver


# Research agent tools
@tool
def search_web(query: str) -> str:
    """Search the web for information.

    Args:
        query: What to search for.
    """
    search = TavilySearch(max_results=2, include_answer=True)
    result = search.invoke({"query": query})
    return str(result.get("answer", "No results"))


# Writing tools
@tool
def format_report(title: str, sections: list) -> str:
    """Format a report with title and sections.

    Args:
        title: Report title.
        sections: List of section dictionaries with 'heading' and 'content'.
    """
    report = f"# {title}\n\n"
    for section in sections:
        report += f"## {section['heading']}\n{section['content']}\n\n"
    return report


# Create the supervisor agent
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

supervisor = create_agent(
    model=llm,
    tools=[search_web, format_report],
    system_prompt="""You are a research supervisor.
    1. First search the web for information
    2. Then format a clear report from the findings
    Be thorough and organized.""",
    checkpointer=InMemorySaver(),
)

result = supervisor.invoke({
    "messages": [{"role": "user", "content": "Research the latest AI trends in 2026 and write a report."}]
})
print(result["messages"][-1].content)
```

> In the next chapter, you'll learn how to split this into proper subagents.

---

## Try It Yourself

1. Think of a task that needs multiple types of expertise (e.g., "analyze data and write a report")
2. Draw a multi-agent architecture diagram for it (on paper)
3. Identify which agents would have which tools
4. Decide: supervisor, hierarchical, or swarm?

---

## What You Learned

- Why single agents struggle with complex tasks
- Three multi-agent architectures: supervisor, hierarchical, swarm
- When to use each architecture
- How to think about splitting work among agents
- A simple supervisor example

---

## Next Steps

Let's build real subagents that work in parallel using SubAgentMiddleware.

Go to: [26 - Subagents: Parallel Isolated Work](./26-subagents.md)