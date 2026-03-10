import inspect
from functools import wraps
from langsmith import tracing_context
from core.observability import get_traceable, is_tracing_enabled


def traced(name: str, run_type: str = "chain"):
    def _decorator(fn):
        traced_fn = get_traceable()(name=name, run_type=run_type)(fn)
        @wraps(fn)
        def _wrapped(*args, **kwargs):
            if inspect.isgeneratorfunction(fn):
                def generator_wrapper(*args, **kwargs):
                    if is_tracing_enabled():
                        with tracing_context(enabled=True):
                            # Keep tracing context open during iteration
                            yield from traced_fn(*args, **kwargs)
                    else:
                        yield from traced_fn(*args, **kwargs)
                return generator_wrapper(*args, **kwargs)

            else:
                if is_tracing_enabled():
                    with tracing_context(enabled=True):
                        return traced_fn(*args, **kwargs)
                return traced_fn(*args, **kwargs)

        return _wrapped
    return _decorator
