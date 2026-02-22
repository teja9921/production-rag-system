# Medical RAG Chatbot — Engineering Documentation

## Purpose of this document

This document records **architectural decisions, trade-offs, and constraints**
made while building a production-oriented Medical RAG system.

It exists to:
- Preserve decision rationale
- Enable future refactors without guesswork
- Support system design interviews
- Avoid cargo-cult engineering

This is not a tutorial.

---

## 1. Problem Context

We are building a Retrieval-Augmented Generation (RAG) system for
medical / knowledge-heavy documents (PDFs).

Primary constraints:
- Safety-first (no hallucinated answers)
- Deterministic, auditable behavior
- Small dataset initially
- Student-level infrastructure budget
- Interview-grade system design clarity

Failure modes are more expensive than refusal to answer.

---

## 2. Core Design Principles

### 2.1 Control before convenience
No framework is allowed to own critical paths:
- Ingestion
- Chunking
- Retrieval
- Safety gating

Frameworks may wrap these components later.

### 2.2 Refusal is success
The system is designed to **return NO_ANSWER** when confidence is low.
This is intentional and correct behavior.

### 2.3 Determinism over cleverness
Same input must always produce:
- Same chunks
- Same embeddings
- Same retrieval result

---

## 3. Ingestion Layer Decisions

### 3.1 PDF parsing: pypdf (direct)

**Chosen**
- pypdf

**Rejected initially**
- LangChain loaders
- unstructured
- pymupdf4llm

**Reason**
- Page-level control required
- Metadata (page number, doc_id) must be exact
- Silent normalization is unacceptable
- Advanced layout handling deferred to later phase

Current ingestion scope:
- Text-only PDFs
- No OCR
- No images
- No tables

This covers the majority of real-world internal documents.

---

### 3.2 Document identity

- `doc_id` = SHA-256 hash of file bytes
- Guarantees:
  - Stable identity
  - Change detection
  - Idempotent ingestion

Filename-based IDs were rejected.

---

## 4. Chunking Strategy

### 4.1 Custom chunker (character-based)

- Approximation: 1 token ≈ 4 characters
- Fixed-size sliding window
- Fixed overlap
- Deterministic chunk IDs

Chunk ID format:
{doc_id}_p{page_number}_c{chunk_index}


### 4.2 Why not LangChain splitters

Rejected because:
- Hidden tokenization
- Metadata loss
- Non-deterministic behavior across versions

Token-true chunking may be introduced later.

---

## 5. Embedding Layer

### 5.1 Model choice

- `sentence-transformers/all-MiniLM-L6-v2`

Reasons:
- Small
- Fast on CPU
- Stable cosine behavior
- Widely supported

### 5.2 Singleton embedding service

- Model loaded once
- Prevents memory duplication
- Required for API workers later

### 5.3 Normalization

- `normalize_embeddings=True`
- Required for cosine similarity correctness
- Without this, similarity thresholds are meaningless

---

## 6. Vector Store

### 6.1 FAISS (dev-only)

- `IndexFlatIP` with normalized vectors
- Explicit metadata alignment
- Local persistence

Why FAISS now:
- Simple
- Transparent
- Zero magic
- Cheap

Planned later:
- PostgreSQL + pgvector for persistence and auditability

---

## 7. Retrieval & Safety Gating

### 7.1 Top-1 hard gate

Retrieval logic:
1. Retrieve top-k
2. Inspect **top-1 score only**
3. If top-1 < threshold → NO_ANSWER
4. Otherwise include remaining chunks

Reason:
- Prevent weak matches from leaking answers
- Avoid “second-best hallucinations”

### 7.2 Threshold calibration

Threshold is **empirical**, not theoretical.

Observed:
- Small corpora → lower scores
- Generic queries → lower similarity
- Domain mismatch → expected refusal

Default:
- 0.7 (production-safe)
- Temporarily lowered to 0.45 for small dataset validation

---

## 8. Why the system refuses to answer

This is **by design**.

Returning NO_ANSWER is preferable to:
- Hallucinating medical advice
- Providing weakly grounded responses
- Overconfident LLM output

Refusal indicates system integrity.

---

## 9. LangChain & LangGraph Positioning

LangChain is **not removed**.
It is **deliberately delayed**.

### 9.1 What LangChain will do

- Wrap existing retriever
- Orchestrate LLM calls
- Manage flow control
- Improve readability

### 9.2 What LangChain will never do

- Load PDFs
- Split text
- Build vector stores
- Decide safety thresholds
- Hide similarity scores

LangChain is an orchestration layer, not a foundation.

LangGraph will be used to model:
- Conditional execution
- Safety fallback paths
- Multi-step reasoning flows

---

## 10. Current System State

Completed:
- Deterministic ingestion
- Custom chunking
- Embedding service
- FAISS index
- Safety-gated retrieval

Not yet implemented:
- LLM generation
- LangChain wrappers
- LangGraph flow
- API layer
- Authentication
- Persistence
- Monitoring

---

## 11. Interview Positioning

This project demonstrates:
- End-to-end RAG system design
- Safety-first retrieval
- Framework-agnostic engineering
- LangChain used intentionally, not blindly
- Clear trade-off reasoning

This is a **system**, not a demo.

---

## 12. LangChain & LangGraph Integration

LangChain and LangGraph are used strictly as orchestration layers.

- Retriever is wrapped as a LangChain Runnable.
- LangGraph controls conditional execution.
- Core retrieval, thresholds, and safety gates remain custom.

Execution flow:
- If retriever returns ANSWER → LLM node executes.
- If retriever returns NO_ANSWER → graph terminates.

This ensures framework visibility for interviews without sacrificing system control.

## 13. LLM Integration Strategy

The LLM is integrated strictly after retrieval and safety gating.
The system guarantees:
- LLM is never invoked on low-confidence retrieval
- Answers are grounded only in retrieved sources
- Prompts are versioned and externalized

LLM calls are treated as unreliable dependencies, not sources of truth.

---

## 14. Provider Deprecation Handling

Initial LLM integration used Mistral-7B via Hugging Face Inference routing.

During testing, the provider deprecated the endpoint, returning HTTP 410.

Because the system isolates the LLM behind a runnable interface, we replaced the backend with Meta-Llama-3-8B-Instruct without modifying:

- retrieval
- safety logic
- orchestration
- API layer

This validated the architectural goal of model-provider decoupling and operational resilience.

## 15. FastAPI Service Layer (Sprint 2)

We introduced a FastAPI layer to expose the LangGraph-based RAG system as a production-style API.

### Objectives
- Provide a clean HTTP interface for querying the RAG system.
- Preserve strict separation between transport (API) and reasoning (LangGraph).
- Enforce schema validation, error handling, and safety gating.

### Key Endpoints
- POST /query  
  Stateless query interface for RAG execution.

### Design Principles
- No business logic inside FastAPI routes.
- LangGraph remains the single orchestration brain.
- API layer performs only:
  - validation
  - request routing
  - response formatting

### Rationale
Separating orchestration from transport ensures:
- testability
- debuggability
- model and retrieval layer independence
- future compatibility with UI, streaming, and agents

## 16. Persistence Layer & Stateful Chats (Sprint 3)

We introduced a relational persistence layer to enable:

- multi-turn conversations
- chat replay
- auditability
- future agent memory

### Schema Design

Tables:
- users
- conversations
- messages

Each conversation represents one chat thread.  
Messages are strictly ordered and persisted.

### Design Principles

- State lives in the database, not in prompts.
- FastAPI remains stateless.
- Conversation memory is deterministic and replayable.

### API Evolution

New endpoints:
- POST /users
- POST /conversations
- POST /conversations/{id}/query
- GET /conversations/{id}/messages

### Rationale

This design enables:
- resumable chat
- observability
- debugging
- deterministic behavior

and forms the foundation for agentic RAG in later sprints.

## 17. Agentic RAG Architecture (Sprint 4)

We upgraded the system from a stateless RAG pipeline into a stateful, agentic RAG architecture using LangGraph.

### Objectives
- Enable multi-turn conversations
- Introduce persistent memory
- Improve retrieval quality via query rewriting
- Maintain deterministic, explainable reasoning

### Design Principles
- Graph-controlled reasoning (no free-form agent loops)
- Explicit state management
- Deterministic node execution
- Database as the single source of truth for memory

---

### Final Agentic Flow

User Query  
→ Memory Reader  
→ Query Rewriter  
→ Retriever  
→ Answer Generator  
→ Memory Writer  

---

### Graph State Schema

The agent operates over an explicit state object:

- conversation_id  
- query (raw user query)  
- history (conversation memory)  
- rewritten_query  
- retrieved_chunks  
- answer  

This ensures:
- clean separation of concerns  
- debuggability  
- reproducibility  

---

### Memory Strategy

Conversation history is stored in a relational database.

At runtime:
- Memory Reader fetches recent messages
- Summarized context is injected into the query rewriting stage
- Memory Writer persists assistant outputs

This prevents:
- prompt explosion  
- hallucinated context  
- hidden state  

---

### Query Rewriting

A dedicated LLM-based rewriting step expands vague or underspecified queries into retrieval-optimized forms.

Example:

"What is LLM?"  
→ "Define large language models, their purpose, training process, and key characteristics."

This significantly improves retrieval quality while preserving safety gating.

---

### Rationale

Separating memory handling, query rewriting, retrieval, and generation ensures:

- controlled autonomy  
- predictable system behavior  
- production-grade observability  

This architecture supports scalable evolution toward multi-agent systems and advanced reasoning workflows.

## 18. Streaming + UI Integration (Sprint 5)

This sprint introduced real-time streaming and a production-grade user interface.

### Objectives
- Eliminate blocking responses
- Improve perceived latency
- Provide a ChatGPT-style interaction experience
- Enable multi-conversation UX

---

### Streaming Architecture

Two execution paths were introduced:

1. Blocking mode (`/query`)  
   - Full agentic graph  
   - Deterministic execution  
   - Used for debugging and testing  

2. Streaming mode (`/stream`)  
   - Retrieval-only graph  
   - Streaming LLM generation via SSE  
   - Used for UI interaction  

This separation prevents:
- duplicate LLM calls
- inconsistent memory updates
- wasted inference cost

---

### Streaming Data Flow

User Query  
→ Retrieval Graph (memory + rewrite + retrieve)  
→ Streaming LLM  
→ Token-by-token response via SSE  
→ Memory persistence  

This guarantees:
- single LLM invocation
- low latency perception
- real-time feedback  

---

### Streamlit UI Design

The UI was designed as a thin client:

- No business logic
- No orchestration
- No LangGraph usage
- Only FastAPI communication

Features:
- ChatGPT-style interface
- Sidebar conversation switching
- New chat creation
- Persistent session memory
- Real-time streaming display

---

### Rationale

Streaming dramatically improves user experience by:

- eliminating blocking latency
- increasing system responsiveness
- improving perceived intelligence

Separating blocking and streaming pipelines ensures:
- system correctness
- predictable behavior
- production-grade architecture

This sprint completes the transformation from backend AI system into a user-facing AI product.

## 19. Retrieval Intelligence Upgrade (Sprint 6)

### Problem
Initial vector-only retrieval showed:
- Weak recall for short queries
- Poor keyword matching
- Hallucinated answers due to low-relevance context

### Solution Architecture
Semantic Chunking → Hybrid Retrieval (BM25 + FAISS) → Cross-Encoder Reranking → LLM Answering

### Design Decisions
- BM25 added for lexical recall.
- Cross-encoder added to learn true semantic relevance.
- Reranking executed post-retrieval to preserve recall + improve precision.
- Scoring removed from retrieval stage; final relevance computed by reranker.

### Trade-offs
- Increased latency (~100–300ms per query).
- Higher CPU usage.
- Major improvement in answer quality and grounding.

### Outcome
- Significant retrieval precision improvement.
- Hallucination reduction.
- More consistent citation grounding.

## 20. Production Hardening & Architecture Finalization (Sprint 7)

### Goal

- Stabilize conversation lifecycle, finalize reasoning architecture, harden API + UI for production behavior, and add observability (logging + latency).

### Task 7.1 — Persistent User & Conversation State

#### What was done

- Implemented cookie-based user identity using encrypted cookies.
- Ensured user identity persists across UI refreshes.
- Backend user creation made idempotent.
- Conversations restored reliably on reload.

#### Why

- Prevented duplicate users/sessions.
- Enabled ChatGPT-like continuity.

### Task 7.2 — Conversation Management UX

#### What was done

- Sidebar conversation list with:
- Auto-generated titles
- Manual rename
- Regenerate title
- Delete conversation
- Graceful handling of untitled / newly created chats.
- Active conversation highlighting.

#### Why

- Matches real-world chat product UX.
- Clean separation of UI state vs backend truth.

### Task 7.3 — Reasoning Graph Refactor

#### What was done
- Replaced multiple graphs with a single reasoning graph:
- Optional query rewrite
- Hybrid retrieval (dense + BM25)
- Reranking
- Moved memory read/write out of graphs into API endpoints.
- Streaming and non-streaming endpoints now share the same reasoning core.

#### Why
- Eliminated double DB writes.
- Reduced unnecessary LLM calls.
- Simplified mental model and debugging.
- Industry-aligned architecture (graphs = reasoning only).

### Task 7.4 — Production Hardening & Observability

#### What was done

- Added structured logging at:
- Request start
- Request end
- Failure points
- Added latency metrics:
- Total request latency
- Streaming TTFB (time to first token)
- Ensured streaming endpoint:
- No per-token logging
- Safe error handling
- API endpoints cleaned and aligned with REST semantics.
- UI aligned with updated API contracts.

#### Why

- Enables debugging under load.
- Makes performance measurable.
- Prevents silent failures in streaming flows.

### Final Outcome of Sprint 7

- Architecture is production-aligned
- UI behavior matches real chat products
- Reasoning layer is clean and reusable
- API is observable and debuggable
- No duplicated side effects or hidden coupling

## 21 — RAG Evaluation + Metrics (Sprint 8)

### Objective

Establish a **deterministic, reproducible evaluation framework** to measure retrieval and answer quality across multiple RAG configurations.

Focus areas:

* Retrieval accuracy
* Ranking effectiveness
* Latency profiling
* Answer correctness (external benchmark)
* Ablation analysis across pipeline variants

---

## T8.1 — GALE Evaluation Dataset (Synthetic Q/A Generation)

Source document:

* *The GALE Encyclopedia of Medicine (Second Edition)*

Pipeline:

1. Semantic chunk extraction
2. Question generation (T5 QG model)
3. QA extraction model
4. Deduplication (regex + semantic similarity)
5. Manual quality filtering

Generation statistics:

* 740 chunks extracted
* 2,683 questions generated
* 1,312 unique questions
* 419 QA pairs extracted
* 389 after global dedupe
* 250 sampled
* **113 high-quality validated evaluation pairs**

Final schema:

* id
* question
* answer (ground truth)
* answer_span (exact text match)
* doc_id
* page_number
* chunk_id
* difficulty

This became the **primary deterministic retrieval benchmark**.

---

## T8.2 — Deterministic Retrieval Evaluation Runner

Implemented evaluation runner that:

Input:

* evaluation_gale_final.json
* active RAG pipeline configuration

Output per query:

* retrieved chunk IDs
* rank position of correct chunk
* hit / miss
* latency
* difficulty

Metrics computed:

* MRR
* Recall@1
* Recall@3
* Recall@5
* latency distribution (p50, p95, mean, max)
* NO_ANSWER rate (when applicable)

All results exported to CSV for aggregation.

---

## T8.3 — Ablation Experiments (Retrieval Config Comparison)

Tested six configurations:

1. hybrid_rerank_v1
2. bm25_v1
3. bm25_rerank_v1
4. hybrid_v1
5. dense_v1
6. dense_rerank_v1

Generated comparative ablation table.

### Key Findings

| Config           | MRR                | Recall@5           | Latency         |
| ---------------- | ------------------ | ------------------ | --------------- |
| hybrid_rerank_v1 | highest            | 1.0                | highest         |
| bm25_v1          | fast               | lower recall       | minimal latency |
| dense_v1         | weakest retrieval  | moderate latency   |                 |
| reranking        | major recall boost | large latency cost |                 |

Conclusion:

* Hybrid + rerank gives best retrieval quality.
* BM25 only gives best speed.
* Reranker dominates recall improvements.
* Retrieval architecture choice = accuracy vs latency trade-off.

This validated the production retrieval design.

---

## T8.4 — MedQuAD Answer Correctness Evaluation

Purpose:
Measure **answer generation quality** against an external medical QA benchmark.

Dataset:

* MedQuAD (sampled subset)

Evaluation method:

* Retrieval + generation pipeline
* ROUGE-L overlap
* semantic similarity (embedding cosine)
* NO_ANSWER rate
* latency

Observed results:

* semantic similarity ≈ 0.67
* ROUGE-L low (expected due to paraphrasing)
* high NO_ANSWER rate (domain mismatch)
* latency consistent with hybrid + rerank cost

Interpretation:
System behaves conservatively when evidence is missing.
Low hallucination risk but limited coverage.

---

## T8.5 — Conversational Stress Test (ruslanmv dataset)

Experiment attempted:

* single-turn doctor response generation

Outcome:

* high hallucination variability
* unstable model availability
* excessive runtime cost
* low evaluation signal relevance

Decision:
**Test discontinued.**
Dataset not aligned with retrieval evaluation goals.

---

## Sprint 8 Deliverables

✔ Synthetic high-quality medical evaluation dataset
✔ Deterministic retrieval evaluator
✔ Metric aggregation pipeline
✔ Full ablation comparison across configurations
✔ External benchmark answer evaluation (MedQuAD)
✔ Performance–latency tradeoff analysis
✔ Evidence-grounded failure detection

---

## Architectural Impact

System is now:

* quantitatively measurable
* reproducibly testable
* configuration comparable
* benchmark validated

Evaluation framework is reusable for future pipeline changes.

---


