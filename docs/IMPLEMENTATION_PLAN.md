# Project Implementation Plan & Milestones

## 1. Milestone Tracking Matrix

| Phase | Description | Key Deliverables | Status |
| :--- | :--- | :--- | :--- |
| **Phase 0** | Repository Inspection | Inspection report, baseline environment audit | **COMPLETED** |
| **Phase 1** | Architecture Documentation | 15 formal specification docs in `docs/` | **COMPLETED** |
| **Phase 2** | Project Scaffolding | `pyproject.toml`, directory layout, `.gitignore`, `.env.example` | **NEXT** |
| **Phase 3** | Configuration & Logging | `app/core/config.py`, `app/core/logging.py`, safety checks | PENDING |
| **Phase 4** | SQLite Persistence | `app/storage/database.py`, DDL schema migrations, repositories | PENDING |
| **Phase 5** | LLM Abstraction & Gemini | `LLMProvider` ABC, `GeminiProvider`, `ResponseValidator` | PENDING |
| **Phase 6** | Character Engine | `character.yaml`, `PersonaEngine`, `PromptBuilder` | PENDING |
| **Phase 7** | Multi-Tier Memory Engine | `MemoryManager`, mathematical scoring, fact extractor | PENDING |
| **Phase 8** | Local RAG Subsystem | FastEmbed ONNX engine, chunker, vectorized SQLite search | PENDING |
| **Phase 9** | User Style Learning | Stylometric tokenizer, confidence calculator, Hinglish detector | PENDING |
| **Phase 10**| Humor Engine | Appropriateness gate, style taxonomy, exemplar retriever | PENDING |
| **Phase 11**| Conversation Orchestration | State machine, chemistry model, end-to-end coordinator | PENDING |
| **Phase 12**| Playwright Browser Agent | Persistent profile manager, `--setup-login` mode, selectors | PENDING |
| **Phase 13**| Message Detection | DOM observer, extraction heuristics, fingerprint idempotency | PENDING |
| **Phase 14**| Response Dispatching | Human-like typing simulation, send confirmation | PENDING |
| **Phase 15**| Reliability & Safety Halt | Checkpoint/CAPTCHA emergency stop, diagnostic dumpers | PENDING |
| **Phase 16**| Testing & Hardening | Unit, integration, browser mock, and fault-injection suites | PENDING |

---

## 2. Phase-by-Phase Task Breakdown & Verification Criteria

### Phase 2: Project Scaffolding
* **Tasks**:
  * Create root `.gitignore`, `.env.example`, `README.md`, `HANDOVER.md`.
  * Create `pyproject.toml` with `uv` declaring dependencies (`playwright`, `pydantic-settings`, `structlog`, `fastembed`, `numpy`, `google-genai`, `pytest`, `pytest-asyncio`, `pyyaml`).
  * Initialize directory layout matching Section 25 modular monolith specification.
* **Verification**: `uv lock` or virtualenv package resolution passes cleanly.

### Phase 3: Configuration & Logging
* **Tasks**:
  * Build `app/core/config.py` with typed Pydantic models for character, LLM, browser, and security settings.
  * Build `app/core/logging.py` using `structlog` with JSON output and secret redaction.
  * Build `app/core/exceptions.py` and `app/core/safety.py`.
* **Verification**: Unit tests confirm environment variable overrides and log formatting.

### Phase 4: SQLite Persistence Layer
* **Tasks**:
  * Implement `app/storage/database.py` with connection pool, WAL mode, and PRAGMA configurations.
  * Create DDL migration applying all 10 schema tables from `docs/DATA_MODEL.md`.
  * Implement repositories: `MessageRepository`, `MemoryRepository`, `StyleRepository`, `ConversationRepository`, `AuditRepository`.
* **Verification**: Integration tests execute CRUD queries and enforce foreign keys and uniqueness constraints.

### Phase 5: LLM Provider Abstraction & Gemini Integration
* **Tasks**:
  * Define `app/ai/provider.py` abstract interface.
  * Implement `app/ai/gemini.py` driver using Gemini 3.8 Flash with structured JSON output and exponential backoff retry.
  * Implement `app/ai/validator.py` executing length, repetition, and prompt-leakage checks.
* **Verification**: Mock tests verify schema compliance and validator rejection logic.

### Phase 6: Character Engine & Prompt Assembly
* **Tasks**:
  * Create `config/character.yaml` with the canonical fictional persona.
  * Implement `app/ai/persona.py` to parse and validate character settings.
  * Implement `app/ai/prompts.py` for deterministic prompt assembly.
* **Verification**: Unit tests verify prompt construction determinism and token budget caps.

### Phase 7: Multi-Tier Memory Engine
* **Tasks**:
  * Implement `app/memory/manager.py` and `models.py`.
  * Implement mathematical scoring formula ($w_s, w_r, w_f$) and recency decay.
* **Verification**: Unit tests verify fact ranking and retrieval thresholds.

### Phase 8: Local RAG Pipeline
* **Tasks**:
  * Implement `app/rag/embeddings.py` using `fastembed` (`bge-small-en-v1.5`).
  * Implement `app/rag/chunker.py` with markdown header awareness.
  * Implement `app/rag/indexer.py` and `retriever.py` with vectorized NumPy dot product.
  * Populate seed character lore in `knowledge/lore.md`.
* **Verification**: Integration tests verify chunk embedding, indexing, and vector similarity retrieval $<25\text{ms}$.

### Phase 9: User Style Learning Subsystem
* **Tasks**:
  * Implement `app/memory/style.py` with stylometric analyzers (sentence length, slang, Hinglish, emojis).
  * Implement statistical evidence count and confidence calculation.
* **Verification**: Unit tests verify style feature extraction across varied test messages.

### Phase 10: Humor Engine
* **Tasks**:
  * Implement `app/humor/engine.py` evaluating emotional distress gating, chemistry modulation, and taxonomy.
  * Implement `app/humor/retrieval.py` querying `humor_examples` table.
* **Verification**: Unit tests confirm humor suppression during serious user messages.

### Phase 11: Conversation Orchestrator & Chemistry
* **Tasks**:
  * Implement `app/conversation/state.py` and `chemistry.py`.
  * Implement `app/conversation/manager.py` unifying all cognitive layers.
* **Verification**: End-to-end integration test with mock LLM verifies full turn lifecycle.

### Phase 12: Playwright Browser Agent Infrastructure
* **Tasks**:
  * Implement `app/browser/session.py` with persistent context management.
  * Implement CLI `--setup-login` mode for interactive human 2FA.
  * Implement `app/browser/selectors.py`.
* **Verification**: Browser launches with persistent user data directory.

### Phase 13 & 14: Message Detection & Dispatching
* **Tasks**:
  * Implement `app/browser/instagram.py` DOM polling and observer.
  * Implement message fingerprinting and idempotency check.
  * Implement human-like keystroke typing cadence simulation.
* **Verification**: Browser mock tests verify message extraction and typing events.

### Phase 15: Reliability & Safety Halt
* **Tasks**:
  * Implement CAPTCHA and checkpoint detector.
  * Implement diagnostic snapshot dumper (DOM HTML + screenshot PNG).
* **Verification**: Mock checkpoint page triggers immediate halt and diagnostic file creation.

### Phase 16: Testing, Hardening & Handover
* **Tasks**:
  * Execute full test suite with coverage reporting.
  * Write `README.md` and `HANDOVER.md`.
* **Verification**: All unit, integration, and browser mock tests pass with 100% clean exit codes.
