# Glossary of Agentic AI Terms

> **Goal:** Plain-English definitions for every term used in this course. Look here before scrolling the docs.
> **Previous chapter:** [Appendix C - Production Directory Structure Guide](./appendix-C-directory-structure.md)
> **Next chapter:** none (this is the final appendix).

---

## How To Read This Glossary

Terms are listed in alphabetical order. The first sentence is the one-line definition. Following lines give context and where in the course the term appears. Code examples use Groq `openai/gpt-oss-120b` throughout.

---

## A

**Agent**
A program that calls an LLM in a loop, decides which tool to call, then acts on the result without a human in the middle. In LangChain 1.x you create one with `create_agent(model, tools, ...)`.

**Agent Loop**
The repeated cycle of "model reasons → tool called → result fed back → model reasons again" until the model returns a final answer with no tool call. The loop is what makes an agent different from a single LLM call.

**AgentState**
The schema passed to `create_agent(state_schema=...)` that augments the default LangGraph state with your own fields (e.g. `notes`, `pending_approvals`). Tools read and write these fields via `ToolRuntime().state`.

**Async**
Code that runs without blocking the main thread (using `async/await`). LangChain agents support both `invoke` and `ainvoke`, streaming versions `stream` and `astream`. Use async when you serve many concurrent users over a web API.

---

## B

**BaseChatModel**
LangChain's base class for chat models. `init_chat_model()` returns one; Groq's `openai/gpt-oss-120b` produces an OpenAI-compatible `BaseChatModel` instance.

**BaseStore**
LangGraph's interface for any long-term key-value memory shared across threads. `InMemoryStore()` and `PostgresStore()` are the two implementations used in this course.

---

## C

**Checkpointer**
A storage object (`InMemorySaver`, `SqliteSaver`, `PostgresSaver`) that saves every state update keyed by `thread_id`. Without one the agent has no short-term memory between turns.

**Command**
The richest return type a tool can produce. Created with `Command(update=..., goto=..., resume=..., interrupt=...)`. Lets a tool write state, route the graph, end the run, or interrupt for human review — all from one returned value.

**Context**
A trusted object passed into the agent by the caller (not the model). Mirrored to tools via `runtime.context`. Use it for things like `user_id`, `db_url`, `environment`, secrets — anything you do not want the model to spoof.

**Context Schema**
The Pydantic class or `TypedDict` passed to `create_agent(context_schema=...)` that defines the shape of `runtime.context`. Enforces types at invoke time so a wrong caller fails fast instead of corrupting downstream tools.

---

## D

**Delta**
The diff produced when comparing two state snapshots in LangGraph. Straming modes such as `"updates"` only emit these deltas, so they are cheaper than full state ("values").

**Dynamic Tools**
Tools whose presence changes per run. Implemented with `ToolListMiddleware`: based on the user prompt, only the relevant subset of tools is presented to the model, keeping context tokens low.

---

## E

**Edge**
In a LangGraph graph, a directed connection from one node to the next. Can be conditional (a function that returns the name of the next node). The agent loop itself is mostly an automatic edge from `agent` → `tools` → `agent`.

**Embeddings**
Numeric vectors (e.g. 768-dimensional float arrays) representing text. Used by RAG to fetch documents whose vectors are close to the query vector. OpenAI's `text-embedding-3-small` and `bge-small-en` are common choices.

---

## F

**FastMCP**
A small Python framework for building an MCP (Model Context Protocol) server. You write Python functions decorated with `@mcp.tool`, and an external agent can call them over stdio or HTTP without copy-pasting code.

**Few-shot**
A prompting technique where the prompt includes example input → output pairs so the model learns the pattern. In agents, few-shot examples often appear in the system prompt for tools that span multiple formats.

---

## G

**Groq**
A cloud inference provider that hosts open-source and partner models with very low latency. We use `openai/gpt-oss-120b` on Groq because the free tier is enough to run every example in this course.

**Guardrails**
Validation middleware that checks user input (and optionally model output) against a producer function. If validation fails, the agent re-prompts the user or refuses the request. Built using the `guardrails` factory or your own middleware.

---

## H

**Harness**
The test-time scaffolding around an agent — the runner that feeds inputs, captures outputs, and asserts behaviour. LangSmith's evaluation harness is the one we use in Chapter 29.

**HITL**
See **Human-in-the-Loop**.

**Human-in-the-Loop (HITL)**
A pattern where the agent pauses execution at a defined checkpoint and waits for a human to approve, edit, or reject the next action. Implemented with `interrupt()` inside a tool and resumed with `Command(resume=...)`.

---

## I

**InMemorySaver**
The simplest LangGraph checkpointer. Stores state in process memory. Perfect for tutorials; lost on every restart, so never use it in production.

**InMemoryStore**
The simplest LangGraph store. A Python dict equivalent, used for long-term memory in dev. Replace with `SqliteStore` or `PostgresStore` for prod.

**Interrupt**
A LangGraph primitive that stops graph execution and resumes only when the caller passes a value via `Command(resume=...)`. The backbone of HITL flows.

---

## L

**LangChain**
The framework that wraps chat models, tools, embeddings, and vector stores into reusable abstractions. Version 1.x introduced `create_agent` and the model-agnostic middleware API used throughout this course.

**LangGraph**
A sub-library of LangChain for orchestrating agents as directed graphs of nodes and edges. `create_agent` is itself a LangGraph graph under the hood.

**LangSmith**
A hosted observability + evaluation platform from LangChain. Set `LANGSMITH_TRACING=true` and every node, tool call, and model call shows up in a trace you can inspect and replay.

**LLM**
Large Language Model. The neural network that takes text in and predicts text out. `openai/gpt-oss-120b` is the LLM this course targets.

---

## M

**MCP**
Model Context Protocol. An open standard for exposing tools, resources, and prompts over a small JSON-RPC channel (stdio or HTTP). Lets an agent call tools written in any language without bundling the tool code.

**Memory**
Two flavours in this course:
- *Short-term* — the conversation inside one `thread_id`, persisted by the checkpointer.
- *Long-term* — cross-thread facts saved in the `store`, usually written by `MemoryMiddleware`.

**Messages**
Structured records exchanged between user, model, and tools. See **HumanMessage**, **AIMessage**, **SystemMessage**, **ToolMessage**.

**Middleware**
Code that wraps the model call (or tool call) and can modify the request, response, or state before it reaches the next stage. Built-ins include `SummarizationMiddleware`, `ModelRetryMiddleware`, `ToolListMiddleware`, etc. Custom middleware is just a callable with the right signature.

**Model**
The LLM the agent calls every loop iteration. Built via `init_chat_model(model="openai/gpt-oss-120b", model_provider="openai", base_url="https://api.groq.com/openai/v1", ...)`.

---

## N

**Node**
A single function in a LangGraph graph. `create_agent` pre-defines two nodes: the `agent` node (calls the model) and the `tools` node (runs tools). Use `name=` on tools to make them discoverable in tracing.

---

## P

**Parallel Tools**
A pattern where the model issues multiple tool calls in one step and the runtime executes them concurrently. Reduces wall-clock latency for independent tool calls. Enabled by default in `create_agent` whenever the model emits multi-call messages.

**Prompt**
The text fed to the model. A system prompt sets behaviour; the user prompt is the actual query; tool descriptions act as mini-prompts each time the model decides which tool to call.

**Prompt Injection**
An attack where untrusted text (e.g. malicious web content the agent is summarising) contains instructions meant to override the system prompt. Defended with input guardrails and tool-name allow-lists.

**Pydantic**
A Python data validation library. LangChain uses Pydantic classes to define tool args schemas, context schemas, and structured output (`response_format`).

---

## R

**RAG**
Retrieval-Augmented Generation. Fetch relevant documents from a vector DB based on the user's query, then inject them into the model's context so the model can answer with grounded citations. Implemented in Chapter 21.

**Response Format**
A parameter on `create_agent(response_format=...)`. Pass `"json"` for raw JSON, a `dict` for JSON-mode schemas, or a Pydantic class for typed structured output. The final agent message conforms to whatever you set.

**Retrieval**
The act of pulling documents from a vector store. Usually `vectorstore.similarity_search(query, k=4)` to get the top-4 closest chunks.

**return_direct**
A `@tool` option that makes the tool's output bypass the final model call. The agent returns the tool's result verbatim. Useful when the tool already produces a perfect user-ready payload (raw JSON, rendered HTML).

---

## S

**Skills**
Markdown files (often with YAML frontmatter) loaded at runtime by `SkillsMiddleware`. They extend an agent's behaviour without touching code — similar to how a human reads a manual before doing a new task.

**State**
A dict-like object that flows through the graph node to node. Contains `messages`, any custom fields you declared in `state_schema`, plus runtime-bookkeeping fields. Every node returns a patch that updates the state.

**State Schema**
The `TypedDict` or Pydantic class passed to `create_agent(state_schema=...)` declaring which custom keys live in the state besides `messages`.

**Store**
Long-term cross-thread memory. `InMemoryStore()` for dev, `SqliteStore` or `PostgresStore` for prod. Tools read/write via `runtime.store.put(namespace, key, value)` and `runtime.store.get(...)`.

**Streaming**
Receiving the agent's output piece by piece as it is produced, instead of waiting for the entire response. Use `agent.stream(..., stream_mode=["messages", "updates"])` for live UIs.

**Stochastic**
Random-by-design. LLMs are stochastic: the same prompt can produce different answers. Lower `temperature` makes them more deterministic; higher `temperature` makes them more creative.

**Subagent**
A secondary agent invoked by a parent agent via `SubAgentMiddleware`. The parent decides *when* to delegate; the subagent owns *how* the delegated task is done and returns a result.

---

## T

**Thread ID**
The unique key for one conversation in short-term memory. Pass it as `config={"configurable": {"thread_id": "user-42"}}` to every `invoke`/`stream` call you want grouped together. Change the key and the agent starts a fresh memory.

**Tool**
A function the model can choose to call. Built via `@tool`. Tools expose their name, description, and argument schema to the model; the model decides which to call and with what arguments.

**Tool Call**
A structured request from the model to invoke a tool by name with given args. Appears inside `AIMessage.tool_calls` and is executed by the `tools` node.

**ToolMessage**
The result of a tool call, paired back to the original call by `tool_call_id` so the model knows which request the result answers.

**ToolRuntime**
The trusted context object available inside a `@tool` function. Access `state`, `context`, `store`, `stream_writer`, `execution_info`, `tool_call_id` via it without passing them as function parameters.

---

## V

**Vector DB**
A database optimised for storing embeddings and finding similar ones by cosine distance. Chroma (used in tutorials) is local; Pinecone, Weaviate, Milvus, pgvector are common production choices.

---

## W

**Writer (stream_writer)**
A callable on `ToolRuntime` that lets a tool emit custom streaming messages mid-execution. Use for progress bars, partial JSON, or live status. Consumed by clients subscribing to `stream_mode="custom"`.

---

## Alphabetical Quick Index

| Term | Section |
|---|---|
| Agent | A |
| Agent Loop | A |
| AgentState | A |
| Async | A |
| BaseChatModel | B |
| BaseStore | B |
| Checkpointer | C |
| Command | C |
| Context | C |
| Context Schema | C |
| Dynamic Tools | D |
| Edge | E |
| Embeddings | E |
| FastMCP | F |
| Few-shot | F |
| Groq | G |
| Guardrails | G |
| HITL / Human-in-the-Loop | H |
| InMemorySaver / InMemoryStore | I |
| Interrupt | I |
| LangChain | L |
| LangGraph | L |
| LangSmith | L |
| LLM | L |
| MCP | M |
| Memory / Messages / Middleware / Model | M |
| Node | N |
| Parallel Tools / Prompt / Prompt Injection / Pydantic | P |
| RAG / Response Format / Retrieval / return_direct | R |
| Skills / State / State Schema / Store / Streaming / Stochastic / Subagent | S |
| Thread ID / Tool / Tool Call / ToolMessage / ToolRuntime | T |
| Vector DB | V |
| Writer | W |

---

> This is the last appendix. Return to the [README](./00-README.md) or jump back to [Appendix C - Production Directory Structure Guide](./appendix-C-directory-structure.md).