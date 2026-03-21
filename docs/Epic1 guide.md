# Epic 1: Foundation & Architecture — Implementation Guide

## Files to Add to Your Project

Copy each file to the exact path shown below in your `agentcoach/` project:

```
agentcoach/
├── docs/
│   └── ARCHITECTURE.md          ← NEW (Story 1.2)
├── src/
│   ├── agents/
│   │   ├── __init__.py          ← (already exists from setup)
│   │   ├── base.py              ← NEW (agent contract)
│   │   └── router.py            ← NEW (Story 1.2 + 6.1)
│   ├── config/
│   │   ├── __init__.py          ← (already exists)
│   │   ├── settings.py          ← (already exists from setup)
│   │   ├── llm_gateway.py       ← (already exists from setup)
│   │   └── prompts.py           ← NEW (centralized prompts)
│   ├── models/
│   │   ├── __init__.py          ← (already exists)
│   │   └── state.py             ← (already exists from setup)
│   ├── graph.py                 ← NEW (LangGraph orchestration)
│   └── main.py                  ← NEW (CLI for testing)
├── tests/
│   ├── test_smoke.py            ← (already exists from setup)
│   └── test_epic1.py            ← NEW (router + graph tests)
├── .env                         ← (already exists)
├── pyproject.toml               ← (already exists)
└── Makefile                     ← (already exists)
```

## Step-by-Step Instructions

### 1. Copy the files
Place each new file in the correct location (paths shown above).

### 2. Install additional dependency
```bash
pip install pydantic-settings
```

### 3. Run tests (NO Azure keys needed)
```bash
pytest tests/test_epic1.py -v
```

You should see ALL tests pass — they use a MockLLMGateway, so no API calls.

Expected output:
```
tests/test_epic1.py::TestRouterClassification::test_valid_json_parsed PASSED
tests/test_epic1.py::TestRouterClassification::test_low_confidence_triggers_clarification PASSED
tests/test_epic1.py::TestRouterClassification::test_malformed_json_falls_back_to_chat PASSED
tests/test_epic1.py::TestRouterClassification::test_markdown_wrapped_json_parsed PASSED
tests/test_epic1.py::TestRouterClassification::test_invalid_mode_defaults_to_chat PASSED
tests/test_epic1.py::TestRouterClassification::test_empty_input_returns_gracefully PASSED
tests/test_epic1.py::TestRouterClassification::test_max_steps_prevents_infinite_loop PASSED
tests/test_epic1.py::TestRouterClassification::test_uses_mini_model_for_classification PASSED
tests/test_epic1.py::TestGraphRouting::test_routes_to_all_agents PASSED
tests/test_epic1.py::TestGraphRouting::test_unknown_agent_defaults_to_chat PASSED
tests/test_epic1.py::TestGraphRouting::test_max_steps_forces_chat PASSED
tests/test_epic1.py::TestStateManagement::test_user_profile_creation PASSED
tests/test_epic1.py::TestStateManagement::test_skill_score_validation PASSED
tests/test_epic1.py::TestStateManagement::test_state_preserves_messages PASSED
tests/test_epic1.py::TestStateManagement::test_valid_modes_are_complete PASSED
tests/test_epic1.py::TestLLMGateway::test_token_counting PASSED
tests/test_epic1.py::TestLLMGateway::test_cost_report_structure PASSED
tests/test_epic1.py::TestGraphIntegration::test_full_flow_greeting PASSED
tests/test_epic1.py::TestGraphIntegration::test_full_flow_interview_request PASSED
```

### 4. Test with real Azure OpenAI (requires .env filled)
```bash
python -m src.main
```

This launches the interactive CLI. Try these messages:
- `"Hi"` → should route to general_chat
- `"I want to prepare for Google ML interviews"` → should route to profile_agent
- `"Let's do a mock interview"` → should route to interviewer_agent
- `"Create a study plan for me"` → should route to planner_agent
- `"What does Amazon ask in ML interviews?"` → should route to strategy_agent
- `"cost"` → shows cost tracking report
- `"state"` → shows current state
- `"quit"` → exits

### 5. Git commit
```bash
git add .
git commit -m "feat(epic-1): foundation — router agent, LangGraph orchestration, architecture doc

- Architecture design doc: Router pattern vs Supervisor, agent inventory, failure modes
- BaseAgent contract: validate → build_prompt → invoke → handle_error lifecycle
- RouterAgent: semantic intent classification with GPT-4o-mini, confidence gating
- LangGraph StateGraph: conditional routing to 5 specialist agents + fallbacks
- Centralized prompts: system prompts + few-shot examples in config module
- CLI entry point for interactive testing
- 19 unit + integration tests with MockLLMGateway (zero API calls)
- Placeholder nodes for Epics 2-5 (incremental integration)"
git push origin main
```

---

## What You Just Built (Interview Talking Points)

### Story 1.1 — Project Scaffold ✅
**Key phrase**: "Production-grade scaffolding with separation of concerns from day one — agents, tools, API, config, and tests in separate packages."

### Story 1.2 — Architecture Design Doc ✅
**Key phrase**: "I authored the architecture design doc before writing any agent code. Router pattern because user intent drives agent selection — unlike my Pre-Auth system where claims follow a fixed pipeline suited to the Supervisor pattern."

### Story 1.3 — LLM Gateway ✅
**Key phrase**: "LLM gateway abstraction with cost-aware routing — GPT-4o for complex reasoning, GPT-4o-mini for classification. Token-level cost tracking per agent per session."

### Story 1.4 — State Schema ✅
**Key phrase**: "Typed state schema with Pydantic validation — every field has a model, not just a dict. Shared state between agents via LangGraph's TypedDict."

### Bonus: Router Agent (Story 6.1 pulled forward)
**Key phrase**: "Semantic intent classification with few-shot prompting, not keyword matching. Confidence-gated routing — below 0.7, we clarify rather than guess wrong."

---

## Architecture Comparison (Memorize This)

| Dimension | Pre-Auth (Healthcare) | AgentCoach (Interview Prep) |
|-----------|----------------------|----------------------------|
| Pattern | Supervisor | Router |
| Routing | Deterministic pipeline | Semantic intent classification |
| State | Per-claim (stateless) | Per-user (persistent) |
| Memory | Short-term only | 3-tier (coming in Epic 2) |
| LLM for routing | N/A (hardcoded) | GPT-4o-mini (few-shot) |
| Agents | 4 (fixed roles) | 5 + Router (dynamic) |

---

## Next: Epic 2 — User Profile & Memory Agent

When you're ready, come back and say **"NEXT — START EPIC 2"**

Epic 2 builds:
- ProfileManager agent (extracts user info from conversation)
- Skill gap assessment engine (LLM-as-judge scoring)
- 3-tier memory architecture (buffer + summaries + vector store)
- Context window optimization (token budget allocator)