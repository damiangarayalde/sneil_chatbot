# Testing Guide

## Quick reference

| Mode | Command | Needs API key? |
|---|---|---|
| Offline (mock LLM) | `LLM_MOCK=true python tests/run__node_or_graph__test.py <bundle.json>` | No |
| Real LLM | `python tests/run__node_or_graph__test.py <bundle.json>` | Yes |
| Dev API smoke test | See section below | Optional |

---

## Offline testing (no OPENAI_API_KEY needed)

Set `LLM_MOCK=true` before running any test bundle. The graph will use
`tests/fixtures/mock_llm/ClassifierOutput.json` and
`tests/fixtures/mock_llm/HandlerOutput.json` instead of calling OpenAI.

### Run individual scenario bundles

```bash
# Classifier node — includes no_llm (heuristic) and llm-tagged scenarios
LLM_MOCK=true python tests/run__node_or_graph__test.py \
    tests/test__node_route_classifier.scenarios.json

# Clarify node
LLM_MOCK=true python tests/run__node_or_graph__test.py \
    tests/test__node_clarify.scenarios.json

# Handoff node
LLM_MOCK=true python tests/run__node_or_graph__test.py \
    tests/test__node_handoff.scenarios.json

# Handler generate node
LLM_MOCK=true python tests/run__node_or_graph__test.py \
    tests/test__node_subgraph__generate.scenarios.json

# Handler retrieve node
LLM_MOCK=true python tests/run__node_or_graph__test.py \
    tests/test__node_subgraph__retrieve.scenarios.json
```

### Run all bundles in one shot

```bash
LLM_MOCK=true python tests/run__node_or_graph__test.py \
    tests/test__node_route_classifier.scenarios.json \
    tests/test__node_clarify.scenarios.json \
    tests/test__node_handoff.scenarios.json \
    tests/test__node_subgraph__generate.scenarios.json \
    tests/test__node_subgraph__retrieve.scenarios.json
```

---

## What the mock covers

The mock returns canned `ClassifierOutput` and `HandlerOutput` responses
matched by keyword substring in `user_text`.

- **Classifier** (`ClassifierOutput.json`) — routes TPMS/AA/CLIMATIZADOR by
  keyword; falls back to a low-confidence response with a clarifying question.
- **Handler** (`HandlerOutput.json`) — detects topic switches and "sigue sin"
  escalation signals; falls back to a generic answer.

Heuristic-only scenarios (tagged `no_llm` in scenario JSON) run identically
in both mock and real mode — no LLM is called at all for those.

### Extending the fixtures

Add entries to the fixture JSON. Order matters — first match wins.
Always keep a fallback entry at the end (no `"match"` key):

```json
[
    {"match": "palabra clave", "output": { ... }},
    {"output": { ... }}
]
```

---

## Full LLM testing (real API)

Leave `LLM_MOCK` unset (or set to `false`). Requires a valid `OPENAI_API_KEY`
in your environment or `.env` file.

```bash
python tests/run__node_or_graph__test.py \
    tests/test__node_route_classifier.scenarios.json
```

Scenarios tagged `llm` will call OpenAI. Results may vary slightly across runs
due to model non-determinism — prefer checking `one_of` expectations for
those cases (already used in the classifier bundle).

---

## Dev API smoke test

Start the server:

```bash
uvicorn app.interfaces.dev_api:app --reload
```

Send a chat message:

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "El sensor TPMS no conecta", "thread_id": "test-001"}' | python -m json.tool
```

With mock LLM (no key needed):

```bash
LLM_MOCK=true uvicorn app.interfaces.dev_api:app --reload
```

---

## Before pushing a PR

Run these checks in order:

```bash
# 1. Offline suite — must pass with zero API calls
LLM_MOCK=true python tests/run__node_or_graph__test.py \
    tests/test__node_route_classifier.scenarios.json \
    tests/test__node_clarify.scenarios.json \
    tests/test__node_handoff.scenarios.json \
    tests/test__node_subgraph__generate.scenarios.json \
    tests/test__node_subgraph__retrieve.scenarios.json

# 2. Full LLM suite — run if you have an API key and changed prompt/chain logic
python tests/run__node_or_graph__test.py \
    tests/test__node_route_classifier.scenarios.json \
    tests/test__node_subgraph__generate.scenarios.json

# 3. Dev API smoke test — verify the graph wires up end-to-end
LLM_MOCK=true uvicorn app.interfaces.dev_api:app &
sleep 2
curl -sf -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "soporte tpms", "thread_id": "smoke-001"}'
```
