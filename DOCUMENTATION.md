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

Initial integration used Mistral via Hugging Face Inference routing.
The provider deprecated the endpoint, returning HTTP 410.

Because the system abstracts the LLM behind a runnable interface,
we swapped to Meta-Llama-3 without modifying:
- Retrieval
- Safety logic
- Orchestration
- Prompts

This validates architectural resilience to provider churn.

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
- future compatibility with UI, streaming, and agents.

## 16. Provider Deprecation Handling

Initial LLM integration used Mistral-7B via Hugging Face Inference routing.

During testing, the provider deprecated the endpoint, returning HTTP 410.

Because the system isolates the LLM behind a runnable interface, we replaced the backend with Meta-Llama-3-8B-Instruct without modifying:

- retrieval
- safety logic
- orchestration
- API layer

This validated the architectural goal of model-provider decoupling and operational resilience.

## 17. Persistence Layer & Stateful Chats (Sprint 3)

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

