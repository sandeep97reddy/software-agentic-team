# Operator Runbook

This runbook provides step-by-step instructions for spinning up the LangGraph framework locally, seeding the workspace, configuring environment variables, and verifying that the pipeline graph compiles perfectly.

## Prerequisites
- Python 3.10+
- Git installed and accessible in the system path
- Docker & Docker Compose (optional, but recommended for state persistence)

---

## 1. Local Environment Setup

Clone the repository and initialize your python virtual environment:

```bash
git clone https://github.com/sandeep97reddy/software-agentic-team.git
cd software-agentic-team

python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

Install the required dependencies using pip (or your preferred package manager):

```bash
pip install -e .
```

## 2. Environment Variables Configuration

The system relies on strict environment variable management for LLM routing and LangSmith observability.

1. Copy the blueprint template:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and configure the following core secrets:
   - `OPENAI_API_KEY`: Required for the core LangChain routing models.
   - `LANGSMITH_API_KEY`: Required to trace execution outputs.

Tracing is automatically enabled internally by `src/core/observability.py` the moment `LANGSMITH_API_KEY` is detected.

## 3. Infrastructure Bootstrapping (Optional)

If you plan on running long, asynchronous jobs and require state persistence (PostgreSQL for LangGraph state check-pointing) and queue mechanics (Redis):

```bash
docker-compose up -d
```

## 4. API Launch & Graph Verification

Launch the FastAPI application server locally:

```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

**What to look for in the startup logs:**
1. You should see `[OBSERVABILITY] LangSmith tracing ENABLED`.
2. The lifespan event in `app.py` triggers `src.core.graph.build_graph()`.
3. Look for the success message: `[OK] LangGraph pipeline compiled successfully`.
4. The server will indicate `Application startup complete`.

## 5. Seeding a Test Project

To trigger the graph, you need to POST a natural language requirement to the orchestration pipeline. We recommend using a simple task for your smoke-test:

```bash
curl -X POST http://localhost:8000/api/v1/execute \
  -H "Content-Type: application/json" \
  -d '{
    "requirements": "Create a python script that outputs the fibonacci sequence up to 10.",
    "project_name": "SmokeTest"
  }'
```

The response will return a UUID (`project_id`). 

## 6. Monitoring Execution Flow

Poll the status endpoint using the UUID returned from the previous step:

```bash
curl "http://localhost:8000/api/v1/status?project_id=<YOUR_PROJECT_ID>"
```

### Trace Inspection
If configured correctly, open your browser and navigate to [LangSmith](https://smith.langchain.com). Under your project (`ai-software-team`), you will see the full distributed trace tree detailing every node transition from `initializer` → `requirement_analyzer` → `architect` and through the execution loop.
