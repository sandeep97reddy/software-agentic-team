# Autonomous AI Software Engineering Team

A production-grade, multi-agent AI system built on LangGraph that plans, codes, tests, and reviews software autonomously in a sandboxed environment.

This project implements a sophisticated orchestration pipeline where specialised AI agents collaborate to break down natural language requirements into architectural designs, write backend and frontend code, run tests, review code for security flaws, and gracefully manage their own token memory.

## Features

- **LangGraph Orchestration Pipeline:** A robust StateGraph managing the flow from Requirements → Architecture → Execution → QA.
- **Specialized AI Agents:** 
  - *Requirement Analyzer:* Parses user inputs into technical specs.
  - *Architect:* Designs folder structures and API schemas.
  - *Task Planner:* Decomposes the architecture into a queue of atomic engineering tasks.
  - *Backend & Frontend Engineers:* Implements specific features using Python/FastAPI and React.
  - *QA Tester:* Auto-generates `pytest` test suites and runs them against new code.
  - *Code Reviewer:* Scans for security vulnerabilities (e.g., SQLi, XSS) and hallucinated dependencies.
- **Secure Tool Layer:** 
  - [FileSystemManager](file:///c:/Users/SANDEEP/Desktop/projects/software%20agentic%20team/src/tools/filesystem.py): Sandboxed read/write capabilities preventing directory traversal.
  - [GitTracker](file:///c:/Users/SANDEEP/Desktop/projects/software%20agentic%20team/src/tools/git_tracker.py): Automatically tracks mutations via local git branches and commits.
  - [SubprocessExecutor](file:///c:/Users/SANDEEP/Desktop/projects/software%20agentic%20team/src/tools/executor.py): Runs external tools (like `pytest`) safely with strict timeouts and allowlists.
- **Anti-Loop Watchdog:** If an agent fails a task (e.g., failing tests or code review) 3 times, the system breaks the infinite loop and redirects to a `Human Approval` node.
- **Memory Compression:** To avoid exceeding LLM context windows on long runs, the system automatically summarizes completed tasks and clears raw execution traces while preserving the core architectural context.
- **LangSmith Observability:** Full distributed tracing of every LLM call, LangGraph node transition, and tool invocation via LangSmith — zero code changes required once `LANGSMITH_API_KEY` is set.
- **FastAPI Endpoints:** Trigger pipeline executions and monitor live task state via standard REST endpoints (`/execute` and `/status`).
- **State Persistence:** Included [docker-compose.yml](file:///c:/Users/SANDEEP/Desktop/projects/software%20agentic%20team/docker-compose.yml) provides PostgreSQL (for LangGraph state check-pointing) and Redis (for task queues).

---

## 📖 Multi-Agent AI Applications

### When to Use What: AI Workflows vs Single Agent vs Multi-Agent

Before building any AI system, it's crucial to understand when and when not to use multi-agents. Generally, it's always best to implement the simplest solution possible to solve a problem.

#### Decision Framework

**1. AI Workflows (Recommended Starting Point)**
*Use when:* The problem can be solved with a well-defined set of steps that you can hardcode.
*Benefits:*
- Fastest path to getting results.
- Easiest to implement, debug, manage, and optimize.
- Most deterministic and repeatable - crucial for production systems reliability.

**2. Single AI Agent**
*Use when:* The problem is open-ended where it's difficult to know beforehand the steps required to solve it. In other words, you can't hardcode the solution.

**3. Multi-Agent System**
*Use when:* You have an open-ended problem that requires an AI agent, AND the problem is complex enough that performance with a single AI agent starts to bottleneck. By splitting the problem across specialized agents (e.g., an Architect, a Developer, a QA Tester), you increase overall reliability and quality.

*Further reading:* [Anthropic Engineering: Building effective agents](https://www.anthropic.com/engineering/multi-agent-research-system)

---

## 🔭 LangSmith Observability

Every LLM call, LangGraph node transition, and tool invocation is automatically traced in [LangSmith](https://smith.langchain.com). This gives you:

- A **full trace tree** of every agent run — see exactly which node called which model with what inputs and outputs.
- **Latency & cost tracking** per node and per pipeline run.
- **Prompt comparisons** across runs to spot regressions.
- **Searchable run history** filterable by `project_id`, `node_name`, and custom tags.

### Setup

1. **Create a free account** at [https://smith.langchain.com](https://smith.langchain.com).
2. Go to **Settings → API Keys** and create a new key.
3. Copy `.env.example` to `.env` and fill in your key:

```bash
cp .env.example .env
```

```dotenv
LANGSMITH_API_KEY=lsv2_your-key-here
LANGSMITH_PROJECT=ai-software-team   # groups all runs in the UI
```

4. Start the server — tracing is **automatically enabled** when the key is present:

```bash
uvicorn src.app:app --reload --port 8000
# Startup log will print:  LangSmith tracing  [ENABLED]
```

### What Gets Traced

| Signal | How |
|---|---|
| Every LLM call (all agents) | Auto-traced by `langchain-core` via env vars |
| Full pipeline run | Tagged with `project_id`, `run_name`, and `node:*` tags |
| Latency per node | Captured automatically by LangGraph's LangSmith integration |
| Custom metadata | `requirements_length`, `node_name` attached to every top-level run |

### Developer Guide: Customizing Traces

The observability implementation resides in [observability.py](file:///c:/Users/SANDEEP/Desktop/projects/software%20agentic%20team/src/core/observability.py):
- **Initialization:** During FastAPI startup (in `src/app.py`), `setup_langsmith()` is invoked. If `LANGSMITH_API_KEY` is present, it dynamically sets LangChain environment variables globally so that all subsequent LLM and LangGraph invocations are automatically intercepted and recorded.
- **Run Configuration & Tags:** When a run is executed in [routes.py](file:///c:/Users/SANDEEP/Desktop/projects/software%20agentic%20team/src/api/routes.py), a custom run configuration is created using the `get_run_config()` helper:
  ```python
  run_config = get_run_config(
      project_id=project_id,
      node_name="pipeline",
      requirements_length=len(body.requirements),
  )
  final_state = _compiled_graph.invoke(initial_state, config=run_config)
  ```
- **Metadata and UI Grouping:** This config attaches:
  - Custom tags like `project:<short_id>` and `node:pipeline`.
  - Metadata attributes such as `project_id`, `node_name`, and any additional key-value pairs (`requirements_length`, etc.).
  - A descriptive `run_name` to group children runs neatly in the LangSmith dashboard.

### Disabling Tracing

Set `LANGCHAIN_TRACING_V2=false` (or simply leave `LANGSMITH_API_KEY` unset). The startup log will confirm `LangSmith tracing [DISABLED]` and no data is sent.

---

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository_url>
   cd <repository_dir>
   ```

2. **Set up the virtual environment**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS / Linux:
   source .venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -e .
   # or if you use Poetry:
   poetry install
   ```

4. **Configure Environment Variables**
   ```bash
   cp .env.example .env
   # Edit .env — at minimum set OPENAI_API_KEY and LANGSMITH_API_KEY
   ```
   See [.env.example](file:///c:/Users/SANDEEP/Desktop/projects/software%20agentic%20team/.env.example) for all available options.

5. **Start Infrastructure (PostgreSQL & Redis)**
   ```bash
   docker-compose up -d
   ```
   Uses the provided [docker-compose.yml](file:///c:/Users/SANDEEP/Desktop/projects/software%20agentic%20team/docker-compose.yml).

## Usage

Start the FastAPI orchestration server defined in [app.py](file:///c:/Users/SANDEEP/Desktop/projects/software%20agentic%20team/src/app.py):
```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/execute` | Trigger a new full pipeline run |
| `GET` | `/api/v1/status?project_id=...` | Poll the current state of a run |
| `POST` | `/api/v1/projects/run` | Alias for `/execute` |
| `GET` | `/api/v1/health` | Liveness probe — returns node list |
| `GET` | `/docs` | Interactive Swagger UI |

**Kick off a new project:**
```bash
curl -X POST http://localhost:8000/api/v1/execute \
  -H "Content-Type: application/json" \
  -d '{
    "requirements": "Build a simple to-do application with a FastAPI backend and React frontend.",
    "project_name": "ToDoApp"
  }'
```

**Check Project Status:**
```bash
curl "http://localhost:8000/api/v1/status?project_id=<UUID_RETURNED_FROM_EXECUTE>"
```

## How It Works Under the Hood

1. **Initialization:** The system creates a temporary sandboxed workspace and initializes a local `git` repository.
2. **Planning Phase:** The [Requirement Analyzer](file:///c:/Users/SANDEEP/Desktop/projects/software%20agentic%20team/src/agents/requirement_analyzer.py) parses user requirements, passing structured specifications to the [Architect](file:///c:/Users/SANDEEP/Desktop/projects/software%20agentic%20team/src/agents/architect.py) (who designs directories and schemas), which then feeds into the [Task Planner](file:///c:/Users/SANDEEP/Desktop/projects/software%20agentic%20team/src/agents/task_planner.py) to build an atomic task queue.
3. **Execution Loop:** 
   - A router pops tasks from the queue and routes them to either the [Backend Engineer](file:///c:/Users/SANDEEP/Desktop/projects/software%20agentic%20team/src/agents/backend_engineer.py) or the [Frontend Engineer](file:///c:/Users/SANDEEP/Desktop/projects/software%20agentic%20team/src/agents/frontend_engineer.py).
   - Engineers write the code, commit it using the git tool, and update the state.
   - A Stagnation Check prevents stuck execution by failing if the code output remains unchanged for two consecutive attempts.
4. **Memory Compression:** As the graph history grows, the [Memory Compression Node](file:///c:/Users/SANDEEP/Desktop/projects/software%20agentic%20team/src/agents/memory.py) condenses completed tasks and logs to keep the LLM context usage minimal.
5. **Validation Phase:** 
   - The [QA Tester](file:///c:/Users/SANDEEP/Desktop/projects/software%20agentic%20team/src/agents/tester.py) generates `pytest` files and executes them.
   - The [Code Reviewer](file:///c:/Users/SANDEEP/Desktop/projects/software%20agentic%20team/src/agents/reviewer.py) scans code artifacts for bugs, security issues, and import anomalies.
   - The [Watchdog](file:///c:/Users/SANDEEP/Desktop/projects/software%20agentic%20team/src/agents/watchdog.py) tracks retry counts. If a task fails three times, it routes control to a human approval node to stop the loop.

