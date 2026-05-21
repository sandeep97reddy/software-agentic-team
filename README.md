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
  - `FileSystemManager`: Sandboxed read/write capabilities preventing directory traversal.
  - `GitTracker`: Automatically tracks mutations via local git branches and commits.
  - `SubprocessExecutor`: Runs external tools (like `pytest`) safely with strict timeouts and allowlists.
- **Anti-Loop Watchdog:** If an agent fails a task (e.g., failing tests or code review) 3 times, the system breaks the infinite loop and redirects to a `Human Approval` node.
- **Memory Compression:** To avoid exceeding LLM context windows on long runs, the system automatically summarizes completed tasks and clears raw execution traces while preserving the core architectural context.
- **FastAPI Endpoints:** Trigger pipeline executions and monitor live task state via standard REST endpoints (`/execute` and `/status`).
- **State Persistence:** Included `docker-compose.yml` provides PostgreSQL (for LangGraph state check-pointing) and Redis (for task queues).

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

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository_url>
   cd <repository_dir>
   ```

2. **Set up the virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install Dependencies**
   If you have Poetry installed:
   ```bash
   poetry install
   ```
   Or via pip (if `requirements.txt` is exported):
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**
   Set up your LLM credentials (e.g., OpenAI API Key, Anthropic API Key) required by the `src.core.config.get_llm()` method.
   ```bash
   export OPENAI_API_KEY="sk-..."
   # or
   export ANTHROPIC_API_KEY="..."
   ```

5. **Start Infrastructure (PostgreSQL & Redis)**
   ```bash
   docker-compose up -d
   ```

## Usage

Start the FastAPI orchestration server:
```bash
uvicorn src.api.routes:router --host 0.0.0.0 --port 8000 --reload
```
*(Note: Wrap the router in a FastAPI app instance in a new `app.py` or modify the command as per your exact setup).*

### Endpoints

**Kick off a new project:**
```bash
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{
    "requirements": "Build a simple to-do application with a FastAPI backend and React frontend.",
    "project_name": "ToDoApp"
  }'
```

**Check Project Status:**
```bash
curl http://localhost:8000/status?project_id=<UUID_RETURNED_FROM_EXECUTE>
```

## How It Works Under the Hood

1. **Initialization:** The system creates a temporary sandboxed workspace and initializes a local `git` repository.
2. **Planning Phase:** The `Requirement Analyzer` passes context to the `Architect`, which passes structural context to the `Task Planner`. A Queue is built.
3. **Execution Loop:** 
   - A router pops tasks from the queue and sends them to either the `Backend` or `Frontend` engineer.
   - The engineers write the code and commit it to the local git repo.
4. **Validation Phase:** 
   - The `QA Tester` auto-generates unit tests and runs them in the sandbox. If tests fail, it re-queues the task with feedback.
   - The `Reviewer` scans the code for security holes.
5. **Memory Management:** If the queue is successfully processed, but the state becomes too large, the `Memory Compression Node` summarizes the log history to save tokens before the next iteration.
