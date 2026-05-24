"""FastAPI entry point with scheduler lifecycle."""
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from loguru import logger

from config import settings
from db.session import engine
from scheduler import create_scheduler
from web.routes import router

# --- Logging setup ---
logger.remove()
logger.add(
    sys.stdout,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{line} | {message}",
    level="INFO",
    serialize=False,
    backtrace=True,
    diagnose=False,
)

# Intercept stdlib logging (APScheduler, SQLAlchemy, httpx)
import logging

class _InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = sys._getframe(6), 6
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
for noisy in ("apscheduler", "sqlalchemy.engine", "httpx", "httpcore"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# --- Sentry (optional) ---
if settings.sentry_dsn:
    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        integrations=[FastApiIntegration(), SqlalchemyIntegration()],
        traces_sample_rate=0.1,
        environment=settings.environment,
        send_default_pii=False,
    )

_scheduler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _scheduler
    logger.info("Starting Polymarket Bot", mode=settings.trading_mode, env=settings.environment)

    if settings.trading_mode == "live":
        from db.session import AsyncSessionLocal
        from services.trader import validate_live_readiness
        async with AsyncSessionLocal() as db:
            await validate_live_readiness(db)
        logger.info("Live readiness check passed")

    _scheduler = create_scheduler()
    _scheduler.start()

    next_weather = _scheduler.get_job("weather_cycle")
    if next_weather:
        logger.info("Next weather cycle", next_run=str(next_weather.next_run_time))

    yield

    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
    await engine.dispose()
    logger.info("Bot stopped")


app = FastAPI(
    title="Polymarket Bot",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)

app.include_router(router)


@app.get("/health")
async def health():
    """Railway health check — no auth required."""
    from db.session import AsyncSessionLocal
    from services.analyzer import health_check as llm_health

    db_ok = False
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(__import__("sqlalchemy").text("SELECT 1"))
            db_ok = True
    except Exception as e:
        logger.warning("DB health check failed", error=str(e))

    sched_ok = _scheduler is not None and _scheduler.running
    llm_ok = await llm_health()

    status = "ok" if all([db_ok, sched_ok, llm_ok]) else "degraded"

    return JSONResponse(
        status_code=200 if status == "ok" else 503,
        content={
            "status": status,
            "db": db_ok,
            "scheduler": sched_ok,
            "llm": llm_ok,
            "mode": settings.trading_mode,
        },
    )
