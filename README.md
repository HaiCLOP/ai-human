# Fictional AI Character — Instagram Browser Agent

A self-hosted, autonomous fictional AI character that communicates through Instagram Web using Playwright. Engineered with modular architecture, strict AI-runtime isolation, local multi-tier memory, stylometric user style adaptation, contextual humor calibration, and local RAG.

---

## Key Capabilities

* **Instagram Web Automation**: Operates via Playwright on the standard web DOM. **Zero Meta API or private API usage**.
* **Strict Anti-Circumvention Policy**: Halts immediately on security checkpoints, CAPTCHAs, or rate-limit modals. Dumps forensic diagnostics (DOM HTML + screenshots) and alerts the operator.
* **Strict AI / Runtime Boundary**: The LLM is an untrusted planner/text synthesizer. It receives zero browser objects, credentials, or filesystem access.
* **Cognitive Subsystems**:
  * **Character Engine**: Decoupled persona, humor, and boundaries externalized in YAML (`config/character.yaml`).
  * **Humor Engine**: Evaluates distress gating, sarcasm calibration, and callback references before authorizing humor.
  * **User Style Learning**: Dynamically detects sentence length, slang, Hinglish, punctuation, and emoji habits with statistical confidence scoring.
  * **Multi-Tier Memory**: Short-term sliding buffer, long-term facts, user preferences, and shared callbacks.
  * **Local RAG**: FastEmbed ONNX local embeddings (`bge-small-en-v1.5`, 384-dim) with vectorized SQLite similarity search.
* **LLM Provider Abstraction**: Hot-swappable AI provider interface with Gemini 3.8 Flash default driver.
* **Persistence & Observability**: SQLite in WAL mode with foreign keys, structured `structlog` JSON logging, and correlation IDs.

---

## Quickstart Guide

### 1. Requirements
* Python 3.11+
* `uv` or `pip`
* Modern Chromium (installed via Playwright)

### 2. Installation
```powershell
# Clone or navigate to the repository
cd e:\Blogs

# Create virtual environment and install dependencies
uv venv
.venv\Scripts\activate
uv pip install -e ".[dev]"

# Install Playwright browser binaries
playwright install chromium
```

### 3. Environment Setup
```powershell
copy .env.example .env
# Edit .env with your GEMINI_API_KEY
```

### 4. Interactive Operator Login (Initial Setup)
```powershell
python -m app.main --setup-login
```
A visible browser window will open. Log in to Instagram, handle any two-factor authentication (2FA) or device confirmations, and press Enter in the terminal. The session is securely saved in `data/browser_profile/`.

### 5. Run the Autonomous Agent
```powershell
python -m app.main
```

### 6. Run Tests
```powershell
uv run pytest -v
```

---

## Architecture Documentation
Detailed specifications are in the [`docs/`](docs/) directory:
* [Architecture Overview](docs/ARCHITECTURE.md)
* [Requirements Specification](docs/REQUIREMENTS.md)
* [Architecture Decision Records (ADR)](docs/ADR.md)
* [Database Schema & Data Model](docs/DATA_MODEL.md)
* [Local RAG Design](docs/RAG_DESIGN.md)
* [Multi-Tier Memory Design](docs/MEMORY_DESIGN.md)
* [User Style Learning Engine](docs/STYLE_LEARNING.md)
* [Character Persona Engine](docs/CHARACTER_ENGINE.md)
* [Humor Engine Specification](docs/HUMOR_ENGINE.md)
* [Browser Agent & Playwright](docs/BROWSER_AGENT.md)
* [Security Boundaries](docs/SECURITY.md)
* [Error Handling & Fault Recovery](docs/ERROR_HANDLING.md)
* [Observability & Structured Logs](docs/OBSERVABILITY.md)
* [Testing Strategy](docs/TESTING.md)
* [Implementation Plan](docs/IMPLEMENTATION_PLAN.md)
