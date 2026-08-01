# AI Infrastructure: The Stack Behind Production Agents

> **Course Navigation:** Previous: [38-reliable-ai-agents.md](./38-reliable-ai-agents.md) | Next: [40-model-serving-and-routing.md](./40-model-serving-and-routing.md)

---

## Why This Lesson Matters

You built an agent. It works in a Jupyter notebook. You show it to your friend — and it crashes because:

- The model API timed out.
- The vector database had no embeddings yet.
- The cache was empty, so you paid for the same LLM call twice.
- The agent task queue was blocked by a failed job.
- You have **no idea** what went wrong because you had no tracing.

**Infrastructure is the difference between "works on my laptop" and "works for real users."**

This lesson gives you the **full map**. Every piece. What it does. Why it exists. How it connects. You don't need to build all of this on day one — but you must know it exists so you can add it as you grow.

---

## The Full Infrastructure Map

Every production AI agent system has **six core infrastructure layers**. Let's look at the whole picture first, then zoom into each piece.

```mermaid
graph TB
    subgraph Layer6["🔍 Layer 6: Observability &amp; Tracing"]
        LANGSMITH["LangSmith / Langfuse<br/>Trace every call"]
        GRAFANA["Grafana<br/>Dashboards &amp; alerts"]
        LANGSMITH --> GRAFANA
    end

    subgraph Layer5["💾 Layer 5: Persistence Store"]
        CHECKPOINT["Checkpoint Store<br/>SQLite → Postgres → Redis<br/>Saves agent state"]
        POSTGRES["Postgres<br/>User data, logs, history"]
    end

    subgraph Layer4["📨 Layer 4: Queue"]
        QUEUE["Message Queue<br/>Redis / RabbitMQ<br/>Async agent tasks"]
        DLQ["Dead Letter Queue<br/>Failed jobs go here"]
    end

    subgraph Layer3["⚡ Layer 3: Cache"]
        SEMCACHE["Semantic Cache<br/>Skip duplicate LLM calls"]
        REDIS["Redis<br/>Tool results + rate limits"]
    end

    subgraph Layer2["🗄️ Layer 2: Vector Database"]
        VECTORS["Chroma / Qdrant<br/>Store embeddings<br/>Semantic search"]
    end

    subgraph Layer1["🧠 Layer 1: Model Serving"]
        GROQ["Groq<br/>openai/gpt-oss-120b<br/>Inference engine"]
        GATEWAY["Model Gateway<br/>Route + fallback<br/>+ rate limit"]
    end

    subgraph Layer0["👤 Layer 0: User"]
        USER["User Request<br/>'Summarize my documents'"]
    end

    USER --> GATEWAY
    GATEWAY --> GROQ
    GROQ --> VECTORS
    VECTORS --> SEMCACHE
    SEMCACHE --> REDIS
    REDIS --> QUEUE
    QUEUE --> DLQ
    QUEUE --> CHECKPOINT
    CHECKPOINT --> POSTGRES
    POSTGRES --> LANGSMITH
    LANGSMITH --> USER

    style Layer0 fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
    style Layer1 fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style Layer2 fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style Layer3 fill:#fce4ec,stroke:#c62828,stroke-width:2px
    style Layer4 fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    style Layer5 fill:#e0f7fa,stroke:#00838f,stroke-width:2px
    style Layer6 fill:#fff9c4,stroke:#f57f17,stroke-width:2px
```

Let's go through each layer, one at a time, in simple words.

---

## Layer 1: Model Serving — The Brain

This is where the LLM actually runs. Your agent sends a prompt, gets a completion back.

**Your setup (and the one we use in this entire course):**

```python
from langchain_groq import ChatGroq

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7,
)
```

**Key decisions at this layer:**

| Decision | Options | Default for beginners |
|----------|---------|----------------------|
| Provider | Groq, OpenAI, Together, Ollama (self-hosted) | **Groq** (free, fast) |
| Model | gpt-oss-120b, llama-3.3-70b, mixtral | **openai/gpt-oss-120b** |
| Temperature | 0.0 = deterministic, 1.0 = creative | 0.7 for chat, 0.0 for tools |

We cover this in depth in the [next lesson](./40-model-serving-and-routing.md).

---

## Layer 2: Vector Database — The Memory

When your agent needs to "remember" documents (PDFs, wikis, past chats), it converts text to numbers called **embeddings** and stores them in a vector database.

```mermaid
flowchart LR
    A["📄 Document<br/>'Cats love warm spots'"] --> B["🔢 Embedding Model<br/>text-embedding-3-small"]
    B --> C[" vector<br/>[0.12, -0.45, 0.89, ...]"]
    C --> D["🗄️ Vector DB<br/>Chroma / Qdrant"]
    E["❓ Query<br/>'Where do cats sleep?'"] --> B
    B --> F["query vector<br/>[0.11, -0.43, 0.85, ...]"]
    F --> D
    D --> G["✅ Top match<br/>'Cats love warm spots'"]

    style A fill:#e8f5e9,stroke:#2e7d32
    style B fill:#fff3e0,stroke:#e65100
    style C fill:#fff3e0,stroke:#e65100
    style D fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
    style E fill:#e8f5e9,stroke:#2e7d32
    style F fill:#fff3e0,stroke:#e65100
    style G fill:#fce4ec,stroke:#c62828
```

**For this course, we use Chroma — it's free and runs locally:**

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    collection_name="my_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db",  # saved to disk
)
```

We cover this in depth in [Lesson 41](./41-vector-db-infrastructure.md).

---

## Layer 3: Cache — The Money Saver

LLM calls cost money (tokens × price). Many calls are **duplicates** — same question, same answer. A cache stores past answers and skips the LLM when it's seen the query before.

There are **two kinds** you need:

1. **Exact-match cache** — same query string, return same response instantly.
2. **Semantic cache** — query *means* the same thing, even if words differ ("What is RAG?" vs "Explain RAG").

```python
from langchain_community.cache import RedisCache
from langchain_core.globals import set_llm_cache

import redis

set_llm_cache(RedisCache(redis.Redis(host="localhost", port=6379)))
```

After this line, your model **automatically** caches. You don't change anything else in your code.

---

## Layer 4: Queue — The Patience Layer

Some agent tasks take 30 seconds. Some take 5 minutes (reading a 200-page PDF). You can't make the user wait on a stuck HTTP request. A **queue** decouples "user asks" from "agent processes."

```mermaid
flowchart LR
    U["👤 User<br/>clicks 'Summarize PDF'"] --> Q["📨 Queue<br/>Redis list"]
    Q --> W1["🤖 Worker 1<br/>processing job"]
    Q --> W2["🤖 Worker 2<br/>processing job"]
    W1 --> S["✅ Job done<br/>notify user"]
    W2 --> DLQ["❌ Dead Letter Queue<br/>retry or inspect"]
    DLQ --> S

    style U fill:#e1f5fe,stroke:#0288d1
    style Q fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    style W1 fill:#fff3e0,stroke:#e65100
    style W2 fill:#fff3e0,stroke:#e65100
    style S fill:#e8f5e9,stroke:#2e7d32
    style DLQ fill:#fce4ec,stroke:#c62828
```

The queue does three jobs:
- **Decouple**: user request returns instantly with a job ID.
- **Scale**: spin up workers as needed.
- **Recover**: failed jobs go to a **dead letter queue (DLQ)** for inspection.

---

## Layer 5: Persistence Store — The Long-Term Memory

Your agent has **state** — what step it's on, what tools it called, what it learned. This state must survive restarts.

LangGraph gives you **checkpointers** for this:

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.postgres import PostgresSaver

# Beginner: in-memory (lost on restart)
checkpointer = MemorySaver()

# Slightly better: SQLite file on disk
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

# Production: Postgres
checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost:5432/agents"
)
```

---

## Layer 6: Tracing — The Debugger's Best Friend

When your agent fails in production, you can't `print()` to debug. You need structured traces — every LLM call, every tool call, every timestamp, every token count.

```python
import os
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "my-agents"
os.environ["LANGSMITH_API_KEY"] = "lsv2_..."
```

This single line makes every LangChain call automatically logged to LangSmith. You open the dashboard and see:

- How many seconds the agent took.
- The full prompt sent to Groq.
- The response.
- Token count and cost.
- Where it failed (if it failed).

**Beginners skip this. Professionals never ship without it.**

---

## Self-Hosted vs Managed: The Eternal Tradeoff

Every layer above has a choice: **build it yourself** or **pay someone to run it**.

```mermaid
flowchart TD
    subgraph Self_Hosted["🔧 Self-Hosted (Free, Hard)"]
        SH1["Ollama on your laptop"]
        SH2["Chroma / Qdrant on a VPS"]
        SH3["Redis on Docker"]
        SH4["Postgres on Railway"]
        SH5["Langfuse (open source)"]
    end
    subgraph Managed["☁️ Managed (Paid, Easy)"]
        M1["Groq API"]
        M2["Pinecone"]
        M3["Upstash Redis"]
        M4["Supabase Postgres"]
        M5["LangSmith"]
    end

    style Self_Hosted fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style Managed fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
```

| Factor | Self-Hosted | Managed |
|--------|-----------|---------|
| Cost | Server cost only (cheap) | Per-request / per-GB (predictable) |
| Setup time | Hours to days | Minutes |
| Maintenance | You patch, you scale | They patch, they scale |
| Data control | 100% yours | On their servers |
| Best for | Learning, sensitive data | Shipping fast, scaling |

**Our course philosophy:** Start fully managed and free (Groq + Chroma local + LangSmith free tier), move to self-hosted only when you must.

---

## Cost Architecture: Where Does the Money Go?

For a typical production agent handling 1,000 queries/day:

```mermaid
pie showData
    title Monthly Cost Breakdown
    "Model inference (Groq)" : 60
    "Vector DB storage" : 15
    "Cache (Redis)" : 8
    "Postgres + queue" : 10
    "Tracing (LangSmith)" : 5
    "Hosting (Railway/Fly)" : 2
```

**The model is always the dominant cost.** Everything else exists to *reduce* model cost (cache) or *retry safely* (queue, store).

---

## A Minimal Stack for Beginners

You don't need all 6 layers on day one. Here's the **smallest production-grade setup** that survives real users:

```python
# -- minimal_stack.py --
import os
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "lsv2_..."

model = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)
checkpointer = MemorySaver()

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

from langchain_core.tools import tool

@tool
def search_docs(query: str) -> str:
    """Search the document store."""
    docs = retriever.invoke(query)
    return "\n".join(d.page_content for d in docs)

agent = create_react_agent(model, [search_docs], checkpointer=checkpointer)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's in my documents?"}]},
    config={"configurable": {"thread_id": "user-1"}},
)
print(result["messages"][-1].content)
```

This is Layers 1 + 2 + 5 + 6. You can ship this. Add cache and queue when traffic grows.

---

## Try It Yourself

1. **Inventory your stack.** List the 6 layers. For each one, write what you currently use (or "none"). Be honest — "none" is fine, the point is to see gaps.

2. **Add tracing.** Set the three `LANGSMITH_*` environment variables above. Run any agent from a previous lesson. Open the LangSmith dashboard. Find your trace. You just added Layer 6.

3. **Draw your own diagram.** Open [mermaid.live](https://mermaid.live). Copy the large flowchart at the top of this lesson. Modify it to show *your* actual stack. What's missing?

4. **Cost gut-check.** Estimate: if your agent runs 100 times/day, and each run uses 2,000 tokens on Groq at $0.10/M tokens, what's your monthly model cost? *(Answer: 100 × 30 × 2000 / 1_000_000 × $0.10 = $0.60/month. Free tier territory. This is why we use Groq.)*

---

## Common Mistakes

- **"I'll add tracing later."** You won't — you'll have a bug at 2 AM with no idea what happened. Add it on day one. The free tier is generous.
- **Mixing dev and prod Chroma paths.** Using `persist_directory="./chroma_db"` in prod means your embeddings get wiped by accident. Use an absolute path and back it up.
- **Treating `MemorySaver` as production storage.** It's in-memory — gone on restart. Switch to SQLite or Postgres before shipping.
- **Forgetting the queue exists.** If a user uploads a 100-page PDF and clicks "summarize," your API will hang for 3 minutes. The browser times out. The user thinks it's broken. Use a queue.
- **Building all 6 layers upfront.** You don't need Postgres + Redis + RabbitMQ on day one. Start with the minimal stack above. Add layers as traffic justifies them.

---

## What You Learned

- Production AI agents rest on **6 infrastructure layers**: model, vector DB, cache, queue, store, tracing.
- Each layer solves a specific failure: model ≠ working, vector DB = memory, cache = cost, queue = patience, store = continuity, tracing = debuggability.
- **Self-hosted vs managed** is a tradeoff between cost and effort — start managed, move self-hosted only when needed.
- The **model is typically 60%+ of cost**; cache and queues exist to reduce that.
- The **minimal shippable stack** is Groq + Chroma + checkpointer + LangSmith. Everything else is added scale.

---

> **Next:** [40-model-serving-and-routing.md](./40-model-serving-and-routing.md) — Deep dive into Layer 1: model serving, routing, fallbacks, and picking providers.