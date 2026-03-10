# Tests Folder Guide

This folder contains lightweight local smoke utilities and unit tests.

## Files kept
- `create_db.py` : create DB tables.
- `test_db.py` : DB CRUD smoke flow.
- `test_conversation.py` : list conversations smoke utility.
- `test_embedder.py` : unit tests for embedder initialization/error paths.
- `test_semantic_chunker.py` : semantic chunking smoke run.

## Files removed as obsolete
- `run_graph.py` : referenced old graph API and missing sample PDF.
- `sample_script.py` : referenced missing sample PDF and old ingestion contract.
- `test_model.py` : contained hardcoded HF token and local-only model assumptions.
- `test_retrieval.py` : outdated retrieval expectations and missing sample PDF.

## Notes
- Smoke scripts are guarded with `if __name__ == "__main__"` to avoid side effects during import.
- `test_embedder.py` is pytest-oriented and safe for CI-style execution.
