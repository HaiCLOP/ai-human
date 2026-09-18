# Architecture Decision Records (ADR)

## ADR-0001: Playwright Browser Automation Over Official & Unofficial APIs
* **Status**: Accepted
* **Context**: Instagram interaction could theoretically be pursued via official Meta Graph APIs, private/reverse-engineered mobile REST endpoints, or browser automation.
* **Decision**: Exclusively use Playwright-driven Chromium browser automation over the standard Instagram Web client.
* **Consequences**:
  * *Positive*: Immune to sudden Meta API deprecations or third-party reverse-engineering breakages; does not require enterprise Meta business verification; operates in natural user context.
  * *Negative*: Subject to DOM layout modifications; requires robust selector fallback mapping and defensive DOM observation.

---

## ADR-0002: Modular Monolith Architecture
* **Status**: Accepted
* **Context**: The system incorporates multiple subsystems (Browser, Conversation Core, Memory, RAG, Humor, Style, AI, Storage). These could be deployed as independent microservices (using message brokers like RabbitMQ or Redis) or as an in-process modular monolith.
* **Decision**: Build the application as an in-process Python modular monolith with strict package boundaries and typed interfaces.
* **Consequences**:
  * *Positive*: Zero infrastructure overhead; no network latency between cognitive modules; single-command startup; trivial local debugging; zero DevOps complexity.
  * *Negative*: Horizontal scaling is not supported out-of-the-box, which is acceptable as the agent is designed for self-hosted, single-account operation.

---

## ADR-0003: SQLite with WAL Mode as the Persistent Datastore
* **Status**: Accepted
* **Context**: The system requires persistence for message logs, idempotency tokens, long-term memory, user styles, and audit trails.
* **Decision**: Adopt SQLite using Write-Ahead Logging (`PRAGMA journal_mode = WAL;`) and enforced Foreign Key constraints (`PRAGMA foreign_keys = ON;`).
* **Consequences**:
  * *Positive*: Zero setup; stored in a single file (`data/agent.db`); atomic ACID transactions; exceptional read concurrency via WAL; robust backup capability.
  * *Negative*: Concurrent high-volume writes across external processes are limited, but well within single-agent DM throughput requirements.

---

## ADR-0004: FastEmbed (ONNX) with Pure NumPy Cosine Similarity for Local RAG
* **Status**: Accepted
* **Context**: The RAG subsystem requires local text embedding generation and vector similarity search without external cloud calls or heavy native C++ compilation burdens on Windows.
* **Decision**: Use `fastembed` with the `BAAI/bge-small-en-v1.5` model (384 dimensions, ONNX Runtime) to generate embeddings, storing raw vectors as binary blobs in SQLite, with retrieval conducted via vectorized NumPy cosine similarity.
* **Consequences**:
  * *Positive*: Fast CPU inference (<15ms per chunk); lightweight dependency footprint (~130MB ONNX model); zero C++ compiler toolchain required on Windows; completely offline.
  * *Negative*: In-memory NumPy cosine calculation is $O(N)$, but with character knowledge bases typically $<10,000$ chunks, search latency remains $<5\text{ms}$.

---

## ADR-0005: Pure Asynchronous Event Loop with Playwright Async API
* **Status**: Accepted
* **Context**: Playwright offers both synchronous and asynchronous Python APIs. The LLM client (Gemini via `google-genai` / HTTPX) and conversation orchestrator operate asynchronously.
* **Decision**: Standardize on `playwright.async_api` and drive the entire application through an `asyncio` event loop.
* **Consequences**:
  * *Positive*: High concurrency without OS thread overhead; non-blocking I/O during LLM generation; clean cooperative multitasking for diagnostic timers and heartbeats.
  * *Negative*: Requires disciplined asynchronous coding (no blocking sleep or synchronous file/network I/O in the event loop).

---

## ADR-0006: Strict AI / Runtime Boundary
* **Status**: Accepted
* **Context**: Providing LLMs with direct tool-execution authority over browser actions poses grave security risks (accidental DM sends, prompt injection vulnerabilities, session destruction).
* **Decision**: Treat the LLM strictly as an offline content generator and planner. The LLM receives sanitized, prompt-assembled text and outputs structured JSON. The runtime controller validates the response and explicitly carries out DOM interactions.
* **Consequences**:
  * *Positive*: Total prevention of browser hijacking via prompt injection; centralized response validation and content safety enforcement.
  * *Negative*: Two-stage pipeline (generate -> validate -> act) introduces minor computational overhead, vastly outweighed by security guarantees.

---

## ADR-0007: Composite Fingerprinting for Message Idempotency
* **Status**: Accepted
* **Context**: Instagram Web DMs dynamically recycle DOM elements without guaranteeing persistent immutable unique message ID attributes across sessions.
* **Decision**: Compute a deterministic composite hash for every detected inbound message:
  $$\text{Fingerprint} = \text{SHA256}(\text{thread\_id} + \text{sender\_handle} + \text{normalized\_text} + \text{time\_bucket\_hour})$$
  Track all processed hashes in a SQLite table with a `UNIQUE` constraint.
* **Consequences**:
  * *Positive*: Guarantees zero duplicate responses even if the browser refreshes or re-renders the chat container.
  * *Negative*: If a user sends the exact same short message twice within the same hour bucket (e.g. "ok"), the second message could be treated as duplicate unless combined with ordinal sequence indices. Ordinal DOM child position is incorporated to disambiguate.

---

## ADR-0008: Externalized YAML Persona Specification
* **Status**: Accepted
* **Context**: Persona, humor boundaries, and communication style parameters should not be hard-coded into prompt templates or application source files.
* **Decision**: Maintain all character personality traits, humor profiles, behavioral preferences, and linguistic guardrails in `config/character.yaml`.
* **Consequences**:
  * *Positive*: Operators can modify persona, adjust humor sliders, or install completely new characters without touching application source code.
  * *Negative*: Requires Pydantic schema validation at application boot to detect syntax or semantic configuration errors early.

---

## ADR-0009: Abstract LLMProvider with Gemini 3.8 Flash Concrete Driver
* **Status**: Accepted
* **Context**: The system must avoid vendor lock-in to any single AI provider while launching with high-performance, cost-effective models.
* **Decision**: Implement an abstract `LLMProvider` interface defining `generate()` and `health_check()`. Implement Gemini 3.8 Flash as the primary provider driver.
* **Consequences**:
  * *Positive*: Zero coupling of core conversation logic to Google SDKs; trivial to add OpenRouter, Ollama, Anthropic, or OpenAI drivers in the future.
  * *Negative*: Features unique to a specific provider (e.g. proprietary search grounding tools) must be abstracted or handled outside the core interface.

---

## ADR-0010: Operator Interactive Login Workflow (`--setup-login`)
* **Status**: Accepted
* **Context**: Instagram enforces aggressive anti-bot detection and 2FA during automated login attempts with raw credentials.
* **Decision**: Do not automate the login form with credentials. Instead, implement a dedicated `--setup-login` CLI mode that opens a visible browser with the persistent profile. The operator logs in manually, completes any 2FA or device approvals, and closes the browser. Subsequent runs use the persisted browser context.
* **Consequences**:
  * *Positive*: 100% compliant with anti-circumvention rules; bypasses automated login detection traps; zero credential exposure in `.env` or application logs.
  * *Negative*: Requires a one-time manual human step when setting up a new profile or after session expiration.
