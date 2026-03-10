# Copilot instructions for `production-rag-system`

This repository is a Python-based medical RAG system with:
- FastAPI backend in `api/`
- retrieval/indexing logic in `rag/`
- orchestration graph/runnables in `orchestration/`
- Streamlit UI in `ui/`
- SQLAlchemy persistence in `db/`

When making changes:

1. Keep edits minimal and scoped to the task.
2. Preserve separation of concerns:
   - API routes should stay in `api/main.py` and delegate business logic.
   - Retrieval and ranking should stay in `rag/`.
   - LLM orchestration should stay in `orchestration/`.
3. Follow existing Python style in touched files (type hints where already used, concise docstrings, no unnecessary comments).
4. Reuse existing configuration from `api/config.py` (`settings`) rather than hardcoding new values.
5. Keep logging consistent with existing structured event style (for example: `event=... | key=value`).
6. Avoid introducing new dependencies unless absolutely required.
7. Do not refactor unrelated modules while fixing a targeted issue.

Validation guidance:
- If tests or checks are available for the touched area, run those targeted checks first.
- For backend changes, verify FastAPI import/runtime path remains valid (`api.main:app`).
