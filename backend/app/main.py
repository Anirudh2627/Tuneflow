"""TuneFlow FastAPI application.

Run locally::

    uvicorn app.main:app --reload --app-dir backend

The ML engine is loaded ONCE during startup (artifacts only -- the API
never trains).
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.api.routers import catalog, events, health, users
from app.core.bootstrap import ensure_ml_on_path
from app.core.config import get_settings
from app.core.logging import get_logger, setup_api_logging
from app.db.base import Base
from app.db.session import SessionLocal, init_engine
from app.services.engine import get_engine_holder

ensure_ml_on_path()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_api_logging(settings.debug)
    init_engine()
    # convenience for local development: create tables when empty DB is used
    # (production deployments should use `alembic upgrade head`)
    from app.db.session import engine as sa_engine

    Base.metadata.create_all(bind=sa_engine)
    get_engine_holder(settings).initialize()
    logger.info("%s v%s started (engine ready=%s)", settings.app_name, settings.version,
                get_engine_holder().is_ready)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description="Personalized music recommendation API: hybrid candidate generation, "
                    "ranking, diversification and explanations.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(users.router)
    app.include_router(catalog.songs_router)
    app.include_router(catalog.artists_router)
    app.include_router(catalog.search_router)
    app.include_router(catalog.genres_router)
    app.include_router(events.events_router)
    app.include_router(events.sessions_router)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    @app.exception_handler(IntegrityError)
    async def integrity_handler(request: Request, exc: IntegrityError):
        if SessionLocal is not None:
            pass
        return JSONResponse(status_code=409, content={"detail": f"Conflict: {exc.orig}"})

    @app.exception_handler(SQLAlchemyError)
    async def sqlalchemy_handler(request: Request, exc: SQLAlchemyError):
        logger.exception("Database error: %s", exc)
        return JSONResponse(status_code=500, content={"detail": "Internal database error"})

    return app


app = create_app()
