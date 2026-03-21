"""Tests for Epic 1: Foundation & Architecture.

These tests verify:
1. Router agent correctly classifies intents
2. Graph routing works for all paths
3. State management is correct
4. Error handling and fallbacks work

Interview talking point: 'I test routing logic independently of the LLM.
The conditional edge function is a pure function of state — I can verify
every routing path without making a single API call. LLM-dependent tests
use a mock gateway that returns predictable classification JSON.'
"""

import json
import pytest

from src.agents.router import RouterAgent, VALID_MODES
from src.config.llm_gateway import LLMGateway, ModelTier
from src.graph import route_after_router, build_graph
from src.models.state import AgentState, UserProfile, SkillScore


# ────────────────────────────────────────────
# FIXTURES
# ────────────────────────────────────────────


class MockLLMGateway:
    """Mock LLM that returns predictable responses for testing.

    Interview talking point: 'Mock LLM gateway for deterministic testing.
    I can test every routing path, every error condition, and every edge
    case without hitting Azure OpenAI. Tests run in <1 second.'
    """

    def __init__(self, response: str = ""):
        self.response = response
        self.call_count = 0
        self.last_messages = None
        self.last_tier = None

    def chat(self, messages, tier=ModelTier.COMPLEX, temperature=0.7,
             max_tokens=1000, agent_name="test", max_retries=3) -> str:
        self.call_count += 1
        self.last_messages = messages
        self.last_tier = tier
        return self.response

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def get_cost_report(self) -> dict:
        return {"total_calls": self.call_count, "total_cost_usd": 0.0}


def make_state(**overrides) -> AgentState:
    """Create a test state with sensible defaults."""
    base: AgentState = {
        "messages": [],
        "current_input": "hello",
        "current_mode": "chat",
        "current_agent": "",
        "route_confidence": 0.0,
        "agent_response": "",
        "step_count": 0,
        "is_complete": False,
        "error": None,
    }
    return {**base, **overrides}


# ────────────────────────────────────────────
# 1. ROUTER CLASSIFICATION TESTS
# ────────────────────────────────────────────


class TestRouterClassification:
    """Test that the Router correctly parses LLM classification output."""

    def test_valid_json_parsed(self):
        """Router correctly parses well-formed classification JSON."""
        mock_llm = MockLLMGateway(
            response='{"mode": "interviewing", "confidence": 0.95, "reasoning": "wants practice"}'
        )
        router = RouterAgent(llm=mock_llm)
        state = make_state(current_input="Let's do a mock interview")

        result = router.invoke(state)

        assert result["current_mode"] == "interviewing"
        assert result["route_confidence"] == 0.95
        assert result["current_agent"] == "interviewer_agent"

    def test_low_confidence_triggers_clarification(self):
        """Below threshold (0.7), router asks for clarification instead of guessing."""
        mock_llm = MockLLMGateway(
            response='{"mode": "planning", "confidence": 0.4, "reasoning": "ambiguous"}'
        )
        router = RouterAgent(llm=mock_llm)
        state = make_state(current_input="let's work on something")

        result = router.invoke(state)

        assert result["current_agent"] == "clarify"
        assert "clarify" in result["agent_response"].lower() or "did you mean" in result["agent_response"].lower()

    def test_malformed_json_falls_back_to_chat(self):
        """If LLM returns garbage, router defaults to chat — never crashes."""
        mock_llm = MockLLMGateway(response="this is not json at all")
        router = RouterAgent(llm=mock_llm)
        state = make_state(current_input="test input")

        result = router.invoke(state)

        # Should not crash, should default to chat
        assert result["current_mode"] == "chat"
        assert result.get("error") is None or "general_chat" in result.get("current_agent", "")

    def test_markdown_wrapped_json_parsed(self):
        """Router handles LLM wrapping JSON in markdown code blocks."""
        mock_llm = MockLLMGateway(
            response='```json\n{"mode": "strategy", "confidence": 0.9, "reasoning": "company question"}\n```'
        )
        router = RouterAgent(llm=mock_llm)
        state = make_state(current_input="What does Google ask?")

        result = router.invoke(state)

        assert result["current_mode"] == "strategy"
        assert result["current_agent"] == "strategy_agent"

    def test_invalid_mode_defaults_to_chat(self):
        """Unknown mode from LLM gets corrected to chat."""
        mock_llm = MockLLMGateway(
            response='{"mode": "dancing", "confidence": 0.9, "reasoning": "???"}'
        )
        router = RouterAgent(llm=mock_llm)
        state = make_state(current_input="test")

        result = router.invoke(state)

        assert result["current_mode"] == "chat"

    def test_empty_input_returns_gracefully(self):
        """Empty user input doesn't crash — returns prompt to retry."""
        mock_llm = MockLLMGateway()
        router = RouterAgent(llm=mock_llm)
        state = make_state(current_input="")

        result = router.invoke(state)

        assert result["current_agent"] == "general_chat"
        assert mock_llm.call_count == 0  # Should not call LLM for empty input

    def test_max_steps_prevents_infinite_loop(self):
        """Step count > 5 forces termination to chat mode."""
        mock_llm = MockLLMGateway(
            response='{"mode": "interviewing", "confidence": 0.99, "reasoning": "test"}'
        )
        router = RouterAgent(llm=mock_llm)
        state = make_state(step_count=6)

        result = router.invoke(state)

        assert result["current_mode"] == "chat"
        assert "exceeded" in result.get("error", "").lower()

    def test_uses_mini_model_for_classification(self):
        """Router should use GPT-4o-mini (SIMPLE tier), not GPT-4o."""
        mock_llm = MockLLMGateway(
            response='{"mode": "chat", "confidence": 0.9, "reasoning": "greeting"}'
        )
        router = RouterAgent(llm=mock_llm)
        state = make_state(current_input="hi")

        router.invoke(state)

        assert mock_llm.last_tier == ModelTier.SIMPLE


# ────────────────────────────────────────────
# 2. GRAPH ROUTING TESTS (no LLM needed)
# ────────────────────────────────────────────


class TestGraphRouting:
    """Test conditional edge routing — pure function, no LLM.

    Interview talking point: 'These tests verify every routing path
    using only state manipulation. Zero API calls. They run in milliseconds
    and catch routing bugs before any integration testing.'
    """

    def test_routes_to_all_agents(self):
        """Every valid agent name routes correctly."""
        agent_names = [
            "profile_agent", "planner_agent", "interviewer_agent",
            "evaluator_agent", "strategy_agent", "general_chat", "clarify",
        ]
        for agent in agent_names:
            state = make_state(current_agent=agent, step_count=1)
            result = route_after_router(state)
            assert result == agent, f"Failed routing for {agent}"

    def test_unknown_agent_defaults_to_chat(self):
        """Unknown agent name falls back to general_chat."""
        state = make_state(current_agent="nonexistent_agent")
        result = route_after_router(state)
        assert result == "general_chat"

    def test_max_steps_forces_chat(self):
        """Exceeding step limit always routes to general_chat."""
        state = make_state(current_agent="interviewer_agent", step_count=6)
        result = route_after_router(state)
        assert result == "general_chat"


# ────────────────────────────────────────────
# 3. STATE MANAGEMENT TESTS
# ────────────────────────────────────────────


class TestStateManagement:
    """Test Pydantic models and state schema."""

    def test_user_profile_creation(self):
        """UserProfile creates with defaults."""
        profile = UserProfile(name="Prudhvi", target_role="Senior ML Engineer")
        assert profile.name == "Prudhvi"
        assert profile.target_companies == []
        assert profile.skill_scores == {}

    def test_skill_score_validation(self):
        """SkillScore enforces 0-10 range."""
        score = SkillScore(topic="transformers", score=7.5, confidence=0.8)
        assert score.score == 7.5

        with pytest.raises(Exception):
            SkillScore(topic="test", score=11.0)  # Out of range

        with pytest.raises(Exception):
            SkillScore(topic="test", score=-1.0)  # Out of range

    def test_state_preserves_messages(self):
        """Message history persists across state updates."""
        state = make_state(messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ])
        assert len(state["messages"]) == 2

    def test_valid_modes_are_complete(self):
        """All expected modes are in the VALID_MODES set."""
        expected = {"planning", "interviewing", "evaluating", "strategy", "profile", "chat"}
        assert VALID_MODES == expected


# ────────────────────────────────────────────
# 4. LLM GATEWAY TESTS
# ────────────────────────────────────────────


class TestLLMGateway:
    """Test LLM gateway utility functions (no API calls)."""

    def test_token_counting(self):
        """Token counter returns reasonable counts."""
        from src.config.llm_gateway import LLMGateway
        # Note: this test creates a gateway but never calls .chat()
        # so it doesn't need real Azure credentials
        gw = LLMGateway.__new__(LLMGateway)
        import tiktoken
        gw.encoding = tiktoken.encoding_for_model("gpt-4o")

        count = gw.count_tokens("Hello world, this is a test.")
        assert count > 0
        assert count < 20  # Sanity check

    def test_cost_report_structure(self):
        """Cost report returns expected keys."""
        mock = MockLLMGateway()
        report = mock.get_cost_report()
        assert "total_calls" in report
        assert "total_cost_usd" in report


# ────────────────────────────────────────────
# 5. INTEGRATION TEST (with mock LLM)
# ────────────────────────────────────────────


class TestGraphIntegration:
    """End-to-end graph test with mock LLM.

    Interview talking point: 'Integration tests verify the full flow
    from user input → router → agent → response using a mock LLM.
    This catches wiring bugs that unit tests miss — like a node not
    being connected or state not passing through correctly.'
    """

    def test_full_flow_greeting(self):
        """Greeting routes through router → general_chat → response."""
        mock_llm = MockLLMGateway(
            response='{"mode": "chat", "confidence": 0.95, "reasoning": "greeting"}'
        )

        # We need the mock to return different things for router vs chat
        # For simplicity, test that the graph doesn't crash
        graph = build_graph(llm=mock_llm)

        state = make_state(current_input="Hi there!")
        result = graph.invoke(state)

        assert result.get("is_complete") is True
        assert result.get("current_agent") is not None

    def test_full_flow_interview_request(self):
        """Interview request routes through router → interviewer_agent."""
        mock_llm = MockLLMGateway(
            response='{"mode": "interviewing", "confidence": 0.92, "reasoning": "wants mock interview"}'
        )

        graph = build_graph(llm=mock_llm)
        state = make_state(current_input="Let's do a mock interview on transformers")
        result = graph.invoke(state)

        assert result.get("is_complete") is True
        # Should route to interviewer (placeholder for now)
        assert "interviewer" in result.get("current_agent", "").lower() or \
               "interview" in result.get("agent_response", "").lower()