"""
app.py — FastAPI application entry-point for the AI Software Engineering Team.

Run locally:
    uvicorn src.app:app --reload --port 8000

Production:
    uvicorn src.app:app --host 0.0.0.0 --port 8000 --workers 4
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router as project_router
from src.core.observability import setup_langsmith

# ──────────────────────────────────────────────────────────────
#  Logging configuration
# ──────────────────────────────────────────────────────────────

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
#  Lifespan  (startup / shutdown hooks)
# ──────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Application lifespan — runs on startup and shutdown."""
    # ── LangSmith observability (must be first) ──────────────
    tracing_on = setup_langsmith()
    tracing_status = "ENABLED" if tracing_on else "DISABLED"

    logger.info("=" * 60)
    logger.info("  [BOT] AI Software Engineering Team -- starting up")
    logger.info("=" * 60)
    logger.info("  Graph compiled               [OK]")
    logger.info("  API routes registered        [OK]")
    logger.info("  LangSmith tracing            [%s]", tracing_status)
    logger.info("  Ready to accept requests")
    logger.info("=" * 60)
    yield
    logger.info("[STOP] Shutting down AI Software Engineering Team")


# ──────────────────────────────────────────────────────────────
#  FastAPI application
# ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Software Engineering Team",
    description=(
        "Production-grade autonomous multi-agent system powered by LangGraph. "
        "Accepts natural-language requirements and orchestrates architect, "
        "developer, tester, and reviewer agents to produce working software."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS (permissive for local dev — lock down in production) ─
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount route handlers ─────────────────────────────────────
app.include_router(project_router, prefix="/api/v1")


# ── Root redirect to docs ────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    """Redirect the bare root to the interactive API docs."""
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url="/docs")


# ──────────────────────────────────────────────────────────────
#  Direct execution  (python -m src.app)
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
    )
