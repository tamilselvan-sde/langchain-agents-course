# Caching and Queue Infrastructure for Agents

> **Course Navigation:** Previous: [41-vector-db-infrastructure.md](./41-vector-db-infrastructure.md) | Next: [43-deployment-platforms.md](./43-deployment-platforms.md)

---

## Why This Lesson Matters

Real agents fail in two ways you don't see in demos:

1. **You pay for the same answer twice.** Ten users ask "what is LangChain?" You call Groq 10 times. You could've answered once and reused.
2. **A long task kills your API.** A user uploads a 50-page PDF and clicks "summarize." Your endpoint hangs for 90 seconds. The browser gives up. The user thinks it's broken.

**Caches** solve problem 1. **Queues** solve problem 2. This lesson shows you how to wire both — with free tools and patterns that work with the Groq `openai/gpt-oss-120b` model we use everywhere.

---

## The Two Layers

```mermaid
flowchart TB
    U["👤 User request"] --> SEM["🔮 Semantic Cache<br/>(skip if seen before)"]
    SEM -->|hit| R1["✅ Return cached response"]
    SEM -->|miss| AGENT["🤖 Agent runs<br/>LLM + tools"]
    AGENT --> TC["🧰 Tool cache (Redis)<br/>skip duplicate tool calls"]
    TC --> STORE["💾 Checkpoint store<br/>SQLite → Postgres → Redis<br/>agent state"]
    AGENT --> Q["📨 Async queue<br/>(if task is long)"]
    Q --> W["⚙️ Worker<br/>runs agent"]
    W -->|done| STORE
    W -->|fail| DLQ["❌ Dead Letter Queue<br/>retries or human review"]
    STORE --> R2["✅ Return response"]

    style U fill:#e1f5fe,stroke:#0288d1
    style SEM fill:#fce4ec,stroke:#c62828,stroke-width:2px
    style AGENT fill:#fff9c4,stroke:#f57f17
    style TC fill:#fce4ec,stroke:#c62828
    style STORE fill:#e0f7fa,stroke:#00838f
    style Q fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    style W fill:#fff3e0,stroke:#e65100
    style DLQ fill:#ffcdd2,stroke:#b71c1c
    style R1 fill:#e8f5e9,stroke:#2e7d32
    style R2 fill:#e8f5e9,stroke:#2e7d32
```

Two layers, two concerns:
- **Cache** = money and time saved on repeat work.
- **Queue** = patience for slow work, retries for failed work.

---

## Layer 1: Semantic Cache — Skip Duplicate LLM Calls

There are two flavors:

| Type | Match on | Cache key | Complexity |
|------|----------|-----------|------------|
| Exact | string-equal prompt | hash(prompt) | Easy |
| **Semantic** | meaning similar | embed(prompt) → cosine search | Smarter, avoids "What's RAG?" vs "Explain RAG?" |

**Semantic caching** is the production-grade choice. Here's how to build one with free tools:

```python
# -- semantic_cache.py --
import os, time
import numpy as np
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
model = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)

cache = []  # list of (prompt_embedding, prompt, response, ts)

def query(prompt: str, threshold: float = 0.95):
    p_vec = np.array(embeddings.embed_query(prompt))
    for vec, p, r, ts in cache:
        sim = float(np.dot(p_vec, vec) / (np.linalg.norm(p_vec) * np.linalg.norm(vec) + 1e-9))
        if sim >= threshold:
            print(f"[CACHE HIT sim={sim:.2f}] matched: {p!r}")
            return r
    response = model.invoke(prompt).content
    cache.append((p_vec, prompt, response, time.time()))
    return response

print(query("What is RAG?"))
print(query("Explain retrieval-augmented generation to me"))  # likely hit
```

Free, no extra service, no setup. Move to a real cache when you scale (see below).

---

## Production Upgrade: LangChain + Redis Semantic Cache

```python
# -- redis_semantic_cache.py --
import redis
from langchain_community.cache import RedisSemanticCache
from langchain_core.globals import set_llm_cache
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq

# 1. Start Redis locally: docker run -p 6379:6379 redis
r = redis.Redis(host="localhost", port=6379)

# 2. Set a global semantic cache (uses Redis + OpenAI embeddings under the hood)
set_llm_cache(RedisSemanticCache(
    redis_client=r,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    distance_threshold=0.05,  # smaller = stricter match
))

# 3. Now any model.invoke() is auto-cached
model = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)

start = time.perf_counter()
out1 = model.invoke("What is RAG?")
t1 = time.perf_counter() - start

start = time.perf_counter()
out2 = model.invoke("Explain retrieval-augmented generation")
t2 = time.perf_counter() - start

print(f"First call: {t1:.2f}s  |  Second (cached): {t2:.4f}s")
print(f"Same answer? {out1.content == out2.content}")
```

Three lines (`set_llm_cache`) and every LLM call in your agent gets cached for free.

---

## Layer 2: Tool Result Cache (Redis)

LLM calls aren't the only expensive things — tool calls are, too. Weather API, web scraper, BLS data, SQL query, RAG lookup — all of these can be cached.

```python
# -- tool_cache.py --
import redis, json, hashlib, time
from langchain_core.tools import tool

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

def cache_key(tool_name: str, args: dict, ttl_bucket: int = 60) -> str:
    raw = f"{tool_name}:{json.dumps(args, sort_keys=True)}:{int(time.time() // ttl_bucket)}"
    return hashlib.sha256(raw.encode()).hexdigest()

@tool
def get_weather(city: str) -> str:
    """Find current weather (mock for demo). Cache 60s."""
    key = cache_key("weather", {"city": city}, ttl_bucket=60)
    cached = r.get(key)
    if cached:
        print(f"[HIT] weather cache for {city}")
        return json.loads(cached)
    print(f"[MISS] fetch fresh weather for {city}")
    val = {"city": city, "temp_c": 22, "weather": "sunny"}
    r.setex(key, 60, json.dumps(val))
    return json.dumps(val)
```

**Bucket TTL idea:** If data changes every minute, cache for 60s. If stable for an hour, cache for 3600s. Tuning TTL is the **#1 lever** for cache usefulness vs correctness.

---

## Layer 3: Message Queue for Async Agent Tasks

When a task takes more than ~3 seconds, the user shouldn't wait on the HTTP request. Send it to a queue. Return a job ID. Worker processes it async. User polls for status.

```mermaid
sequenceDiagram
    participant U as 👤 User
    participant API as 🌐 FastAPI
    participant Q as 📨 Redis queue
    participant W as ⚙️ Worker
    participant S as 💾 Store
    U->>API: POST /summarize {pdf}
    API->>Q: enqueue job {id: 12345}
    API-->>U: 202 Accepted {id: 12345}
    Q->>W: pop job 12345
    W->>S: status=running
    Note over W: ... runs Groq + tools ...
    W->>S: status=done, result=...
    U->>API: GET /jobs/12345
    API->>S: fetch
    API-->>U: 200 result=...
```

**Free option: Redis as a queue (via BRPOP / LPUSH).**

```python
# -- queue_producer.py (your FastAPI route) --
import redis, uuid, json

r = redis.Redis(host="localhost", port=6379)

def enqueue_agent_task(user_query: str) -> str:
    job_id = str(uuid.uuid4())
    r.hset(f"job:{job_id}", mapping={
        "status": "queued",
        "query": user_query,
        "result": "",
    })
    r.rpush("agent_jobs", job_id)
    return job_id

# In your FastAPI handler:
# return JSONResponse({"job_id": enqueue_agent_task(payload)}, status_code=202)
```

```python
# -- queue_worker.py (run separately) --
import redis, time
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

r = redis.Redis(host="localhost", port=6379, decode_responses=True)
model = ChatGroq(model="openai/gpt-oss-120b")

@tool
def echo_tool(s: str) -> str:
    return f"[tool echo] {s}"

agent = create_react_agent(model, [echo_tool])

while True:
    _, job_id = r.brpop("agent_jobs")  # blocks until a job is ready
    job_id = job_id.decode() if isinstance(job_id, bytes) else job_id
    print(f" picked up job {job_id}")
    r.hset(f"job:{job_id}", "status", "running")
    try:
        query = r.hget(f"job:{job_id}", "query")
        result = agent.invoke({"messages": [{"role": "user", "content": query}]})
        final = result["messages"][-1].content
        r.hset(f"job:{job_id}", mapping={"status": "done", "result": final})
        print(f" ✓ job {job_id} done")
    except Exception as e:
        r.hset(f"job:{job_id}", mapping={"status": "failed", "result": str(e)})
        r.rpush("agent_jobs_dlq", job_id)
        print(f" ✗ job {job_id} failed → DLQ")
```

Run the worker as a separate process. Run the API as another. They talk through Redis. The user gets an instant `202 Accepted` even for 30-second tasks.

---

## Layer 4: Checkpoint Store Backends

LangGraph saves agent state via **checkpointers**. Each option ships in `langgraph.checkpoint.X`:

```mermaid
flowchart LR
    subgraph Beginner["Beginner"]
        M["MemorySaver<br/>(in-memory, lost on restart)"]
    end
    subgraph Small["Single prod"]
        S["SqliteSaver<br/>(file on disk, OK for 1 process)"]
    end
    subgraph Prod["Production"]
        P["PostgresSaver<br/>(durable, multi-process)"]
        R["RedisSaver<br/>(low-latency, ephemeral)"]
    end

    M -->|"upgrade"| S -->|"upgrade"| P
    P -->|"for low-latency"| R

    style M fill:#e1f5fe,stroke:#0288d1
    style S fill:#e8f5e9,stroke:#2e7d32
    style P fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style R fill:#f3e5f5,stroke:#6a1b9a
```

```python
# -- checkpoints.py --
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq

# 1. Beginner: vanishes on restart
checkpointer = MemorySaver()

# 2. Slightly bigger: persists to file
checkpointer = SqliteSaver.from_conn_string("./checkpoints.db")

# 3. Production: durable, multi-process, multi-instance
checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost:5432/agents"
)

# 4. Production ultra-low latency: Redis (ephemeral, fast check-ins)
# from langgraph.checkpoint.redis import RedisSaver
# checkpointer = RedisSaver(redis.Redis(host="localhost", port=6379))

agent = create_react_agent(
    ChatGroq(model="openai/gpt-oss-120b"),
    [],
    checkpointer=checkpointer,
)

# Each user = a thread_id
config = {"configurable": {"thread_id": "user-1"}}
r1 = agent.invoke({"messages": [{"role": "user", "content": "Hi, I'm Anu."}]}, config=config)
print(r1["messages"][-1].content)

r2 = agent.invoke({"messages": [{"role": "user", "content": "What's my name?"}]}, config=config)
print(r2["messages"][-1].content)  # "Anu" — checkpoint restored it!
```

---

## Layer 5: Dead Letter Queue (DLQ) for Failed Agent Runs

A job that fails 5 times in a row is probably broken (bad prompt, bad tool, bad data). Retry-blindly only makes things worse — you can poison your LLM costs.

**Solution:** failed jobs go to a DLQ. A human or a "healer" worker inspects them later.

```python
# -- dlq.py --
import redis, time, json
from langchain_groq import ChatGroq

r = redis.Redis(host="localhost", port=6379, decode_responses=True)
model = ChatGroq(model="openai/gpt-oss-120b")

MAX_RETRIES = 3

def run_job(job_id: str):
    query = r.hget(f"job:{job_id}", "query")
    retries = int(r.hget(f"job:{job_id}", "retries") or 0)
    try:
        # pretend this can fail; simulate 50% chance of failing for demo
        if hash(query) % 2 == 0:
            raise RuntimeError("simulate failure")
        out = model.invoke(query).content
        r.hset(f"job:{job_id}", mapping={"status": "done", "result": out})
        return
    except Exception as e:
        retries += 1
        r.hset(f"job:{job_id}", mapping={
            "retries": retries,
            "last_error": str(e),
        })
        if retries >= MAX_RETRIES:
            r.hset(f"job:{job_id}", "status", "dead")
            r.rpush("agent_jobs_dlq", json.dumps({
                "job_id": job_id,
                "error": str(e),
                "query": query,
            }))
            print(f" 💀 {job_id} sent to DLQ")
        else:
            r.rpush("agent_jobs", job_id)   # re-queue for retry
            print(f" ⏳ {job_id} retry #{retries}")
```

**Healer pattern:** a separate worker pops DLQ messages, asks the LLM "look at this failing query, propose a fix", and either rewrites the prompt (stores it) or escalates to a Slack notification.

---

## Putting It All Together

```python
# -- full_stack.py (concept skeleton -- run each part in its own process) --
import redis, json
from langchain_groq import ChatGroq
from langchain_core.globals import set_llm_cache
from langchain_community.cache import RedisCache
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.checkpoint.sqlite import SqliteSaver

# 1. Cache layer: every LLM call cached in Redis
set_llm_cache(RedisCache(redis.Redis(host="localhost", port=6379)))

# 2. Checkpointer: SQLite file (upgrade to Postgres in prod)
checkpointer = SqliteSaver.from_conn_string("./agent_checkpoints.db")

# 3. Tool with own cache
r = redis.Redis(host="localhost", port=6379, decode_responses=True)

@tool
def current_weather(city: str) -> str:
    """Get weather. Cached 60s in Redis."""
    key = f"weather:{city.lower()}:{int(time.time() // 60)}"
    cached = r.get(key)
    if cached:
        return cached
    val = json.dumps({"city": city, "temp_c": 22})
    r.setex(key, 60, val)
    return val

# 4. Build agent with both cache (auto) and checkpointer
from langgraph.prebuilt import create_react_agent
agent = create_react_agent(
    ChatGroq(model="openai/gpt-oss-120b", temperature=0.7),
    [current_weather],
    checkpointer=checkpointer,
)

# 5. Run inside queue worker (see queue_worker.py)
print(agent.invoke(
    {"messages": [{"role": "user", "content": "Weather in Chennai?"}]},
    config={"configurable": {"thread_id": "user-1"}},
)["messages"][-1].content)
```

This is seconds away from production-grade: cache layer free (Redis), tool cache free (Redis), checkpoints durable (SQLite → Postgres), queue async (Redis), DLQ isolated (Redis).

---

## Try It Yourself

1. **Add semantic cache locally.** Run `semantic_cache.py` above. Ask "what is langchain?" and "tell me about langchain the framework". Confirm second call hits the cache and returns instant.

2. **Build a queue.** Start Redis (`docker run -p 6379:6379 redis`). Run `queue_worker.py` in one terminal. Run a Python script that calls `enqueue_agent_task("Summarize Python in 3 bullets.")` 5 times. Watch the worker pick them up.

3. **Simulate DLQ.** In `dlq.py`, set `MAX_RETRIES=2`. Enqueue a job with a query that always fails (e.g., `query = ""` which raises). Confirm after 2 attempts it goes to DLQ. Inspect `agent_jobs_dlq` with `redis-cli`.

4. **Checkpoint migration.** Build an agent with `MemorySaver`. Restart the script. Show the agent loses memory. Switch to `SqliteSaver`. Restart. Show the agent remembers the previous turn.

---

## Common Mistakes

- **Caching everything.** If a tool returns real-time data (stock price), don't cache it like the weather example — or set TTL = 1s and add a "fresh" flag.
- **Setting threshold too loose for semantic cache.** `distance_threshold=0.5` matches unrelated queries. Start at `0.05` (very strict). Loosen only when you measure hits.
- **Forgetting cache invalidation on data update.** If your product catalog changes, your tool cache must be invalidated — or stale prices will leak to users.
- **Sync polling for job status.** Polling `GET /jobs/12345` every 200ms hammers your server. Use SSE / WebSocket, or poll with exponential backoff.
- **Treating `MemorySaver` as durable.** It's not. Restart = data gone. Switch to SQLite before shipping anything real.
- **No DLQ.** Failed jobs vanish. You'll never find out *why*. Always add a DLQ even if it just logs to a file.
- **One Redis for everything.** Keyspaces collide. Use prefixes (`cache:`, `tool:`, `checkpoint:`, `job:`, `dlq:`) or separate Redis databases 0–15.

---

## What You Learned

- Two infra layers protect against cost and patience: **caches** and **queues**.
- **Semantic cache** (Redis + embeddings) makes "what is RAG?" and "explain RAG" share an answer.
- **Tool caches** via Redis key+TTL pattern are easy and pay off immediately.
- **Async queues** turn a 30-second HTTP call into a 200ms ack + job_id + later fetch.
- **Checkpointers** save agent state — MemorySaver (dev) → SQLite (small prod) → Postgres (production) → Redis (ultra-low-latency).
- **DLQ** catches failed jobs after N retries so a human (or a healer agent) can investigate.
- Two processes (API + worker) + Redis is enough to start a production agent.

---

> **Next:** [43-deployment-platforms.md](./43-deployment-platforms.md) — Where do agents actually run? Compare LangGraph Platform, FastAPI + Docker, Railway, Fly.io, Render, self-hosted Kubernetes.