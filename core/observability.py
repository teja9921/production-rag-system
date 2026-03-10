from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from api.config import settings
from core.logger import get_logger

logger = get_logger("core.observability")

@dataclass 
class TracingState:
    enabled: bool = False 
    traceable: Optional[Callable] = None

_state = TracingState()

def is_tracing_enabled() -> bool:
    return _state.enabled

def init_observability()-> None:
    if not settings.LANGSMITH_TRACING:
        logger.info("event=TRACING_DISABLED")
        return
    
    if not settings.LANGSMITH_API_KEY:
        raise RuntimeError("LANGSMITH_TRACING=true but LANGSMITH_API_KEY is missing")
    if not settings.LANGSMITH_PROJECT:
        raise RuntimeError("LANGSMITH_TRACING=true but LANGSMITH_PROJECT is missing")

    try:
        from langsmith import Client, traceable as ls_traceable
    except Exception as exc:
        raise RuntimeError("LangSmith package is not installed") from exc

    client_kwargs = {"api_key": settings.LANGSMITH_API_KEY}
    if settings.LANGSMITH_ENDPOINT:
        client_kwargs["api_url"] = settings.LANGSMITH_ENDPOINT

    client = Client(**client_kwargs)

    def configured_traceable(*, name: str, run_type: str = "chain"):
        return ls_traceable(
            name = name,
            run_type= run_type,
            project_name= settings.LANGSMITH_PROJECT,
            client= client
        )
    
    _state.enabled = True
    _state.traceable = configured_traceable
    logger.info("event=TRACING_ENABLED | project=%s", settings.LANGSMITH_PROJECT)


def get_traceable():
    if _state.traceable is not None:
        return _state.traceable
    
    def noop_traceable(*, name: str, run_type: str = "chain"):
        def _decorator(fn):
            return fn
        return _decorator
    
    return noop_traceable