# Monitoring Utilities

Shared Layer 3 utilities:
- `experiment_registry.py` : persist eval metrics by dataset/config.
- `extract_gale_failures.py` : generate GALE failure question list.
- `build_failure_index.py` : build FAISS index from failure list.
- `failure_similarity.py` : runtime query similarity checker.

Module usage examples:
- `python -m evaluation.monitoring.extract_gale_failures`
- `python -m evaluation.monitoring.build_failure_index`
