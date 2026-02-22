GALE evaluation set (auto-generated)

Source:
- The GALE Encyclopedia of Medicine (Second Edition)

Generation:
- Question Generation: valhalla/t5-small-qa-qg-hl
- Extractive validation: deepset/roberta-base-squad2
- Filters: chunk-size, keyword, QA confidence, question heuristics
- Deduplication: TF-IDF cosine similarity

Files:
- evaluation_gale.json : final QA pairs
- sample_for_review.json : small sample for human review

Recommended usage:
- Use these for recall@k and grounding/faithfulness tests only.
- Do manual spot-check before large-scale runs.
