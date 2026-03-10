from functools import wraps
from core.observability import get_traceable, is_tracing_enabled

def traced(name: str, run_type: str = "chain"):
    def _decorator(fn):
        @wraps(fn)
        def _wrapped(*args, **kwargs):
            traced_fn = get_traceable()(name=name, run_type=run_type)(fn)
            if is_tracing_enabled():
                from langsmith import tracing_context
                with tracing_context(enabled=True):
                    return traced_fn(*args, **kwargs)
            return traced_fn(*args, **kwargs)
        return _wrapped
    return _decorator
