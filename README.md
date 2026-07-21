# SmartDesk

Minimal LangGraph CLI with a supervisor and three specialist subgraphs:
ResearchAgent, TaskAgent, and WriterAgent. The project contains exactly the nine
tools required by the problem statement.

The hybrid supervisor uses structured LLM intent classification, deterministic
safety guards, and a regex fallback; it reviews specialist results, controls
handoffs/fan-in, and alone produces the final user-visible answer.

## Setup and run

Use Python 3.11 or 3.12. Python 3.14 currently triggers compatibility warnings in
LangChain's Pydantic compatibility layer and is not recommended for this project.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:OPENROUTER_API_KEY="..."
$env:TAVILY_API_KEY="..."
python graph.py
```

The CLI loads `OPENROUTER_API_KEY` and `TAVILY_API_KEY` from `.env` (or from
already-set environment variables).

Enter a unique username. The CLI automatically uses that user's stable `main`
thread, so restarting resumes it while different users remain isolated. Press
Enter on an empty message to exit. Every invocation uses a recursion limit of 20.

`build_graph()` returns a compiled graph supporting both `invoke()` and
`stream()`; the intentionally minimal CLI uses `invoke()` only.

## Architecture

```text
Supervisor Plan -> Memory (trim every turn; semantic search for non-chat)
  -> Chat | ResearchAgent | TaskAgent | WriterAgent | parallel Research + Task
  -> Supervisor Review per specialist
  -> Error | WriterAgent handoff | Supervisor Final
  -> Save extracted memories -> END
```

Parallel ResearchAgent and TaskAgent branches are reviewed independently, then
joined exactly once. Research/Task/Join may hand off to WriterAgent; every
successful path ends at supervisor-owned synthesis. Any node/review failure
routes to a graceful Error node and then END.

Research supports HTML and PDF sources. Persistent long-term memory uses
`sentence-transformers/all-MiniLM-L6-v2` embeddings for semantic top-3 retrieval.
The model downloads once from Hugging Face and then loads from the local cache.
Strict msgpack deserialization is enabled. Security pins are
`langgraph==1.2.9` (>=1.0.10) and `langgraph-checkpoint-sqlite==3.1.0`
(>=3.0.1). See `architecture.png` for the compiled flow.

## Three-turn example

```text
You: Research LangGraph checkpointing and save a note.
SmartDesk: [sourced summary]
You: Write a short report from that research.
SmartDesk: [report using the research handoff]
You: Create a task with review and publish steps.
SmartDesk: [persisted ordered task]
```
