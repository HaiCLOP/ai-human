# Operational Handover & Maintenance Guide

## 1. System Overview
The **Instagram AI Character Agent** is a production-engineered, local modular monolith. It communicates with Instagram users via Instagram Web DMs using Playwright. It operates strictly within explicit security boundaries and ethical AI constraints.

---

## 2. Directory Layout & Module Responsibilities

| Path | Purpose |
| :--- | :--- |
| `app/core/` | Pydantic configuration, structured `structlog` logging, exception definitions, safety filters. |
| `app/storage/` | SQLite database manager (`PRAGMA WAL`), DDL migrations, and transactional repositories. |
| `app/ai/` | Abstract `LLMProvider`, Gemini 3.8 Flash driver, prompt builder, persona engine, response validator. |
| `app/memory/` | Multi-tier memory manager, hybrid scoring, fact extraction, stylometric learning. |
| `app/rag/` | FastEmbed local ONNX embedding generation, markdown chunker, vectorized SQLite search. |
| `app/humor/` | Humor appropriateness classifier, humor taxonomy, callback tracking, exemplar retriever. |
| `app/conversation/` | Orchestration manager, conversation state machine, conversational chemistry dynamics. |
| `app/browser/` | Playwright session manager, DOM selector registry, typing simulator, diagnostic dumpers. |
| `app/scheduler/` | Autonomous background jobs (housekeeping, RAG re-indexing, memory consolidation). |
| `config/` | `character.yaml` declaring persona traits, humor parameters, and behavioral guidelines. |
| `knowledge/` | Fictional lore, backstory, and operator-supplied knowledge files for local RAG. |
| `docs/` | Comprehensive technical architecture and design specifications. |
| `tests/` | Unit, integration, and browser mock test suites. |

---

## 3. Operational Runbooks

### Runbook 1: Initial Account Provisioning & Setup Login
1. Ensure `.env` is created from `.env.example` and contains a valid `GEMINI_API_KEY`.
2. Run: `python -m app.main --setup-login`
3. A visible Chromium window will open at `https://www.instagram.com/direct/inbox/`.
4. Log in using your Instagram account. Complete SMS or App-based 2FA if prompted.
5. Once your DM inbox is clearly visible, return to the terminal and press `Enter`.
6. The session cookies and tokens are written to `data/browser_profile/`.

### Runbook 2: Daily Autonomous Operation
* Start the agent: `python -m app.main`
* The agent loads the cached browser profile and begins monitoring for inbound messages.
* To run in headed mode for visual observation, set `BROWSER_HEADLESS=false` in `.env`.

### Runbook 3: Responding to a Security Challenge Halt
* If Instagram displays an anti-bot checkpoint or CAPTCHA, the agent halts immediately:
  * Exit code: `101` (`EXIT_CHALLENGE_DETECTED`).
  * Telemetry screenshot saved to: `logs/diagnostics/challenge_screenshot_*.png`.
  * Telemetry DOM saved to: `logs/diagnostics/challenge_dom_*.html`.
* **Remediation**:
  1. Inspect the diagnostic screenshot to understand the nature of the challenge.
  2. Run `python -m app.main --setup-login` to manually complete the verification as a human operator.
  3. Resume normal autonomous operation.

### Runbook 4: Updating DOM Selectors When Instagram UI Shifts
* Instagram periodically updates class names and layout containers.
* All selectors are centralized in `app/browser/selectors.py`.
* To update:
  1. Inspect the new DOM elements in Chromium DevTools.
  2. Add the new CSS/ARIA selector to the top of the relevant list in `SELECTORS`.
  3. Run `pytest tests/browser` to verify.

### Runbook 5: Modifying Character Persona or Humor
* Edit `config/character.yaml`.
* Adjust traits (`playfulness`, `sarcasm`, `warmth`), humor boundaries, or preferred behavior.
* No code recompilation or rebuild is necessary; changes are loaded on next agent start.
