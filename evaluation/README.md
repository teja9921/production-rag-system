# Evaluation Folder Structure

This directory is organized by dataset and evaluation purpose.

## Dataset folders
- `gale/` : GALE dataset, scripts, outputs, and failure artifacts.
- `medquad/` : MedQuAD evaluation scripts and outputs.
- `chat/` : chat evaluation scripts and outputs.
- `myset/` : template/onboarding scaffold for new datasets.

## Ablation folders
- `chunking_ablation/`
- `reranker_ablation/`
- `rewrite_ablation/`

## Monitoring core (Layer 3)
- `monitoring/experiment_registry.py`
- `monitoring/extract_gale_failures.py`
- `monitoring/build_failure_index.py`
- `monitoring/failure_similarity.py`

These are shared utilities used by API runtime and evaluation scripts.

## Important generated artifacts
- `evaluation/experiment_registry.json` (created when metrics are registered)
- `<dataset>/outputs/...` (eval outputs)
- `<dataset>/<dataset>_failure.index` (failure similarity index)

## Cleanup policy applied
- Removed all `__pycache__/` directories under `evaluation/`.
- Removed empty `evaluation/results/` directory.
