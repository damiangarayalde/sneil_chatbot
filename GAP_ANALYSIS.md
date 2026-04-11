# Gap Analysis: sneil_chatbot vs. Production-Grade WhatsApp Chatbot

> **Scope:** Core bot logic, testability, and resilience — not WhatsApp Meta API integration (deferred) and not evals (deferred).

---

## Current State (What's Working Well)

| Strength | Detail |
|---|---|
| LangGraph state machine | Clean multi-route pipeline with routing edges, precheck, classifier, and per-route handler subgraphs |
| Structured LLM outputs | Classifier and handler both use Pydantic models — prevents hallucinated JSON |
| Heuristic fast-paths | Keyword routing, low-info detection, and retrieval decisions avoid unnecessary LLM calls |
| Scenario-based testing | JSON scenario bundles cover routing, RAG, generation, clarification, and handoff nodes |
| SQLite checkpointing | Multi-turn context persists across turns via LangGraph's `SqliteSaver` |
| Dev API + HTML mockup | FastAPI `/chat` endpoint with WhatsApp-style UI shows state internals (confidence, route, attempts) |
| YAML config | Per-route thresholds, keyword lists, and prompt paths are externalized |

---

## Gap 1 — No Offline / Mockable LLM Layer

**Problem:** `init_llm()` in `app/core/utils.py` always instantiates `ChatOpenAI`. Every test that touches the classifier or handler nodes makes a real API call. You cannot run the scenario test suite without an active `OPENAI_API_KEY`, and a broken LLM response crashes nodes with no recovery path.

**Proposed refactor:**
- Introduce a `LLMProvider` abstraction (a thin wrapper around `llm.invoke()`).
- Ship a `MockLLMProvider` that reads fixtures from JSON files and returns canned `ClassifierOutput` / `HandlerOutput` responses keyed by scenario ID or message hash.
- Inject the provider at graph-build time via a config flag (`LLM_MOCK=true`).
- This makes the entire scenario test suite runnable with zero API calls.

**Files to change:** `app/core/utils.py`, `app/core/graph/build.py`, `app/core/graph/route_classifier/nodes.py`, `app/core/graph/route_handler/factory_and_nodes.py`

**Testable without WhatsApp:** Yes — all scenario tests immediately benefit.

---

## Gap 2 — No Retry / Timeout on LLM Calls

**Problem:** LLM calls are bare `chain.invoke()` calls with no timeout, no retry, and no circuit breaker. A network hiccup or OpenAI rate-limit error propagates as an unhandled exception straight through the graph and up to the API caller.

**Proposed refactor:**
- Wrap all LLM invocations with `tenacity` retry logic: exponential backoff, max 3 attempts, retry on `openai.RateLimitError` and `openai.APIConnectionError`.
- Add a configurable `LLM_TIMEOUT_S` env var (default 15s) using `asyncio.wait_for` or `httpx` client timeout.
- On final failure, emit a structured fallback: return a `ClassifierOutput(estimated_route="UNKNOWN", confidence=0.0)` or the equivalent handler fallback so the graph can route to handoff gracefully.

**Files to change:** `app/core/utils.py` (wrap init), or a new `app/core/llm_client.py`

**Testable without WhatsApp:** Yes — the dev API `/chat` endpoint will no longer 500 on transient LLM errors.

---

## Gap 3 — No Structured Logging / Request Tracing

**Problem:** All observability is `print()` statements. There is no request ID, no log levels, no JSON output, and no way to correlate a conversation turn across nodes. In production this makes debugging nearly impossible.

**Proposed refactor:**
- Replace all `print()` calls with Python's `logging` module configured at startup (level from `LOG_LEVEL` env var).
- Add a `request_id` (UUID) to each `/chat` call; thread it through the LangGraph config dict so every node log line carries it.
- Emit structured JSON lines (one dict per log event) containing: `request_id`, `thread_id`, `node`, `duration_ms`, `route`, `confidence`.
- The flow-logging wrapper in `app/core/graph/flow_logging.py` already intercepts every node — just change its output format.

**Files to change:** `app/core/graph/flow_logging.py`, `app/interfaces/dev_api.py`, startup config

**Testable without WhatsApp:** Yes — visible immediately in dev API logs.

---

## Gap 4 — Session Has No Expiry or Context Window

**Problem:** `SqliteSaver` accumulates messages forever per `thread_id`. A long-running conversation will grow the prompt beyond the model's context limit. There is also no cleanup job, so the DB grows unbounded.

**Proposed refactor:**
- Add a `MAX_HISTORY_MESSAGES` config (e.g. 20 messages). In the `end_of_turn` node, truncate `state["messages"]` to the last N, always keeping the system prompt.
- Add a `THREAD_TTL_HOURS` config. A lightweight background task (or startup check) deletes threads whose last checkpoint is older than TTL.
- Expose a `/health` endpoint that also reports DB size and active thread count.

**Files to change:** `app/core/graph/nodes.py` (end_of_turn), `app/core/persistence.py`, `app/interfaces/dev_api.py`

**Testable without WhatsApp:** Yes — verifiable via the dev API by counting messages returned in state.

---

## Gap 5 — Dev API is Synchronous and Has No Auth

**Problem:** `graph.invoke()` is a blocking call. Under any concurrent load the API will queue requests behind each other, and a slow LLM call blocks the entire event loop. The dev API also has only an optional `TEST_KEY` header check — no enforcement by default.

**Proposed refactor:**
- Switch `graph.invoke` → `await graph.ainvoke` and make the route handler `async`.
- Add FastAPI `BackgroundTasks` or a simple `asyncio.Queue` so the API can return a `202 Accepted` with a job ID and the client polls or uses SSE for the result. (This mirrors how real WhatsApp webhooks work — you ACK immediately and process async.)
- Enforce `TEST_KEY` unconditionally when set, and add a startup warning if it is unset.

**Files to change:** `app/interfaces/dev_api.py`, graph build entry points

**Testable without WhatsApp:** Yes — testable with `httpx` or `curl` against the dev API.

---

## Gap 6 — Tool Use is Hardcoded, Not LLM-Driven

**Problem:** Catalog lookup and RAG retrieval are triggered by hardcoded keyword lists in Python, not by the LLM. This works for known terms but breaks silently for synonyms, misspellings, or new product names. Adding a new trigger requires a code change.

**Proposed refactor:**
- Define catalog lookup and RAG retrieval as proper LangChain `@tool` functions with typed input/output schemas.
- Bind them to the handler LLM via `llm.bind_tools([catalog_tool, rag_tool])`.
- Keep the existing heuristics as a fast-path pre-check (cheap, no LLM), but let the LLM invoke the tools when heuristics don't match.
- This is purely internal — no WhatsApp changes needed.

**Files to change:** `app/core/tools/catalog_tool.py`, `app/core/tools/rag.py`, `app/core/graph/route_handler/factory_and_nodes.py`

**Testable without WhatsApp:** Yes — scenario tests for the generate node can assert on tool calls.

---

## Gap 7 — Custom Test Runner Instead of pytest

**Problem:** `tests/run__node_or_graph__test.py` is a hand-rolled test framework with its own assertion logic, `print()`-based output, and `SystemExit` for failures. It works, but it has no integration with the standard Python test ecosystem: no `-k` filtering by scenario ID or tag, no markers to separate fast/slow or LLM/no-LLM tests, no coverage reporting, no `pre-commit` or CI hooks, and no way to run a single failing scenario in isolation.

Each scenario JSON already has `id`, `title`, and `tags` fields — the data is pytest-ready. The scenario JSON files themselves do not need to change.

**Proposed refactor:**

**1. Add pytest to `pyproject.toml`:**
```toml
[project.optional-dependencies]
test = ["pytest>=8.0", "pytest-asyncio>=0.24", "pytest-cov>=6.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
markers = [
  "unit: pure Python, no LLM, no I/O",
  "integration: node-level, may call LLM",
  "e2e: full graph, multi-turn",
  "no_llm: scenario uses only heuristics — safe offline",
  "llm: scenario requires a real OpenAI call",
]
```

**2. Move shared helpers to `tests/conftest.py`:**
Cut `_apply_patch`, `_matches_expect`, `_assert_expect`, `_last_ai_message`, and `_resolve_node_fn` from `run__node_or_graph__test.py` and expose them as pytest fixtures/helpers. The logic is good — just relocating it.

**3. Restructure into three test layers:**
```
tests/
  conftest.py                      # shared helpers + scenario runner logic
  unit/
    test_heuristics.py             # is_low_info, asked_for_human, direct_route_from_keywords
    test_config_schema.py          # config YAML parses without validation errors
  integration/
    conftest.py                    # pytest_generate_tests: parametrizes from JSON scenario files
    test_classifier_node.py        # wraps test__node_route_classifier.scenarios.json
    test_handler_nodes.py          # wraps test__node_subgraph__*.scenarios.json
    test_clarify_node.py           # wraps test__node_clarify.scenarios.json
    test_handoff_node.py           # wraps test__node_handoff.scenarios.json
    test_route_subgraph.py         # wraps test__route_subgraph.scenarios.json
  e2e/
    fixtures/                      # multi-turn full-graph conversation JSON fixtures
    test_full_graph.py             # full graph.invoke(), all turns, final state assert
```

The `integration/conftest.py` reads the `"tags"` array from each scenario JSON and auto-applies `@pytest.mark.no_llm` or `@pytest.mark.llm` accordingly — no manual annotation needed.

**4. Add a `Makefile` with tiered targets:**
```makefile
test-fast:   # offline, no OPENAI_API_KEY needed — run before every edit
    pytest -m "unit or no_llm" --tb=short

test:        # full integration suite — run before a commit
    pytest -m "not e2e" --tb=short

test-all:    # before a push
    pytest --tb=short --cov=app --cov-report=term-missing
```

**What `run__node_or_graph__test.py` becomes:** Either kept as a convenience CLI (`python run_test.py bundle.json`) for ad-hoc exploration, or deleted once pytest is working. It is not the ground truth anymore.

**Key unit tests that don't exist today** (Gap 7a — `msg_heuristics_no_llm.py` has zero test coverage despite being the most logic-dense pure-Python file in the project):
- `is_low_info`: ~10 parametrized cases (empty, short, greetings, meaningful text)
- `asked_for_human`: regex coverage for all `_HUMAN_REQUEST_PATTERNS`
- `direct_route_from_keywords`: single-route match, multi-route ambiguity, no support/sale word

**Files to add:** `tests/conftest.py`, `tests/unit/test_heuristics.py`, `tests/unit/test_config_schema.py`, `tests/integration/conftest.py`, one thin test file per existing scenario JSON, `tests/e2e/`, `Makefile`

**Files to change:** `pyproject.toml` (add test deps + pytest config)

**Testable without WhatsApp:** Yes — the `unit` and `no_llm` tiers are fully offline.

---

## Gap 8 — Config Has No Validation or Schema

**Problem:** `config.yaml` is loaded with `yaml.safe_load()` and keys are accessed by string with no validation. A typo in a threshold value or a missing `prompt_file` path silently produces `None` and may cause a cryptic error deep in a node.

**Proposed refactor:**
- Define a `Pydantic` model for the config schema: `AppConfig`, `RouteConfig`, `ClassifierConfig`.
- Parse the YAML through the model at startup. Hard-fail on startup (not at first request) if required keys are missing or values are out of range.
- This catches misconfiguration in dev before a message is ever sent.

**Files to change:** `app/core/config/` (add `schema.py`), startup in `build.py` or `dev_api.py`

**Testable without WhatsApp:** Yes — a unit test can load the config and assert no validation errors.

---

## Gap 9 — Incomplete Route Prompts

**Problem:** Several routes (`GENKI`, `CARJACK`, `MAYORISTA`, `CALDERA`) have empty or near-empty prompt files. The classifier can route to these routes but the handler has nothing to work with, causing the LLM to hallucinate or return empty answers.

**Proposed refactor:**
- Add a startup check: for each route defined in `config.yaml`, verify that the `prompt_file` path exists and is non-empty.
- Emit a warning (not a hard failure) for routes with prompts shorter than a minimum threshold.
- As a stopgap, add a generic `fallback_handler_prompt.md` that any underdeveloped route can inherit.

**Files to change:** `app/core/prompts/routes/`, `app/core/graph/build.py` (startup checks)

**Testable without WhatsApp:** Yes — a startup smoke test.

---

## Gap 10 — No WhatsApp Webhook Adapter (Deferred, But Design It Now)

**Problem:** The current `dev_api.py` is the only HTTP interface. When the Meta API is eventually connected, the webhook payload format is completely different from `{"text": "...", "thread_id": "..."}`. Building the adapter later risks breaking the core pipeline to accommodate it.

**Proposed refactor (design only, implement when Meta is ready):**
- Add `app/interfaces/whatsapp_webhook.py` with a `POST /webhook` endpoint.
- Keep payload parsing (extracting `phone_number_id`, `from`, `text.body`) in the adapter layer only — the graph receives the same `HumanMessage` regardless of source.
- Map `from` (phone number) to `thread_id` in the adapter.
- Add `GET /webhook` for Meta's token verification handshake.
- The dev API `/chat` and WhatsApp webhook become two adapters over the same graph — nothing in the core changes.

**Files to add:** `app/interfaces/whatsapp_webhook.py`

**Testable without WhatsApp:** Yes — unit-testable with a mock Meta payload fixture.

---

## Prioritized Execution Order

| Priority | Gap | Effort | Unlocks |
|---|---|---|---|
| 1 | Gap 1 — Mock LLM layer | Medium | All scenario tests run offline |
| 2 | Gap 7 — Migrate to pytest + add unit tests | Low-Medium | Offline test gate, coverage, CI-ready |
| 3 | Gap 8 — Config validation | Low | Fail-fast on misconfiguration |
| 4 | Gap 3 — Structured logging | Low | Debuggability in dev API |
| 5 | Gap 2 — Retry + timeout | Medium | Resilience without WhatsApp |
| 6 | Gap 4 — Session expiry | Low | Prevents prompt overflow |
| 7 | Gap 5 — Async dev API | Medium | Concurrency + ACK pattern |
| 8 | Gap 6 — LLM tool use | High | Scalable catalog/RAG triggers |
| 9 | Gap 9 — Prompt completeness | Content work | Route correctness |
| 10 | Gap 10 — Webhook adapter design | Low design / Medium impl | WhatsApp readiness |
