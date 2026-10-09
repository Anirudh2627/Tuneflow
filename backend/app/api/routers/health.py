"""Health & engine status router."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.services.engine import EngineHolder
from app.api.deps import dependency_engine

router = APIRouter(tags=["health"])


@router.get("/health")
def health(engine: EngineHolder = Depends(dependency_engine)) -> dict:
    return {
        "status": "ok",
        "engine": "hybrid" if engine.is_ready else "degraded-popularity-fallback",
        "components": {
            name: comp is not None for name, comp in engine.components.items()
        } if engine.components else {},
        "degraded_reason": engine.degraded_reason,
    }
