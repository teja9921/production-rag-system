# MySet Evaluation Workflow

## Required inputs
1. Corpus file(s): `data/myset_source.pdf`
2. QA eval set: `evaluation/myset/evaluation_myset_final.json`

Expected eval JSON row fields:
- `id`
- `question`
- `chunk_id`
- optional: `difficulty`

## Run order
1. `python -m evaluation.myset.scripts.run_eval_ranked`
2. `python -m evaluation.myset.scripts.compute_metrics`
3. `python -m evaluation.myset.extract_failures`
4. `python -m evaluation.myset.build_failure_index`

After step 4, restart API if runtime similarity checker should use `myset` index.

## Output files
- Retrieval eval CSV:
  - `evaluation/myset/outputs/eval_outputs/myset_hybrid_rerank.csv`
- Metrics JSON:
  - `evaluation/myset/outputs/eval_outputs/metrics_myset_hybrid_rerank_v1.json`
- Failure questions:
  - `evaluation/myset/myset_failures.json`
- Failure FAISS index:
  - `evaluation/myset/myset_failure.index`

