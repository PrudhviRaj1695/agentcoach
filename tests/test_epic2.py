"""Tests for Epic 2: User Profile & Memory Agent.

These tests verify:
1. Bayesian skill score updates work correctly
2. Profile extraction and CRUD operations
3. Three-tier memory retrieval
4. Token budget allocation
5. SQLite persistence layer
6. ProfileManager agent flow

All tests run WITHOUT Azure OpenAI keys (mock LLM).

Interview talking point: 'I test the Bayesian update algorithm with
deterministic inputs — verifying that high-confidence scores are stable
and low-confidence scores are volatile. These are the invariants that
matter for a coaching system.'
"""

import os
import pytest
import tempfile

from src.models.state import (
    AgentState,
    AssessmentSource,
    SkillCategory,
    SkillScore,
    UserProfile,
    InterviewSession,
)
from src.memory.memory_manager import (
    ConversationBuffer,
    SessionSummaryStore,
    SemanticMemory,
    MemoryManager,
)
from src.memory.token_budget import TokenBudgetAllocator
from src.memory.persistence import ProfileStore


# ── Reuse mock from Epic 1 ──

class MockLLMGateway:
    def __init__(self, response: str = "Mock response"):
        self.response = response
        self.call_count = 0

    def chat(self, messages, tier=None, temperature=0.7,
             max_tokens=1000, agent_name="test", max_retries=3) -> str:
        self.call_count += 1
        return self.response

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def get_cost_report(self) -> dict:
        return {"total_calls": self.call_count, "total_cost_usd": 0.0}


# ════════════════════════════════════════════════════════════
# 1. BAYESIAN SKILL SCORE TESTS
# ════════════════════════════════════════════════════════════

class TestSkillScore:
    """Test Bayesian skill update algorithm.

    These tests verify the CORE INVARIANT: high-confidence scores
    should be stable, low-confidence scores should be volatile.
    """

    def test_initial_score_creation(self):
        """New skill starts at specified score with low confidence."""
        score = SkillScore(
            topic="transformers",
            category=SkillCategory.DEEP_LEARNING,
            score=5.0,
            confidence=0.1,
        )
        assert score.score == 5.0
        assert score.confidence == 0.1
        assert score.assessment_count == 0

    def test_bayesian_update_low_confidence(self):
        """Low-confidence score shifts significantly on new evidence.

        User has score=5.0 with confidence=0.1 (1 assessment).
        New evidence: scored 9.0 on a diagnostic.
        Expected: score shifts substantially toward 9.0.
        """
        score = SkillScore(
            topic="transformers",
            category=SkillCategory.DEEP_LEARNING,
            score=5.0,
            confidence=0.1,
        )
        score.update_score(9.0, AssessmentSource.DIAGNOSTIC)

        # Score should shift significantly toward 9.0
        assert score.score > 6.0, "Low-confidence score should shift toward new evidence"
        assert score.confidence > 0.1, "Confidence should increase after assessment"
        assert score.assessment_count == 1

    def test_bayesian_update_high_confidence(self):
        """High-confidence score is STABLE against contrary evidence.

        THE KEY INVARIANT: A user with 10 assessments at ~7.0 should NOT
        drop to 4.0 because of one bad answer.
        """
        score = SkillScore(
            topic="transformers",
            category=SkillCategory.DEEP_LEARNING,
            score=7.0,
            confidence=0.9,  # High confidence (many assessments)
            assessment_count=9,
        )
        score.update_score(3.0, AssessmentSource.SINGLE_QA)

        # Score should NOT crash to 3.0
        assert score.score > 5.5, f"High-confidence score shouldn't crash: got {score.score}"
        assert score.score < 7.0, "Score should decrease somewhat"

    def test_score_history_tracking(self):
        """Every update appends to score history for trend analysis."""
        score = SkillScore(
            topic="sql",
            category=SkillCategory.SQL_CODING,
            score=5.0,
        )
        score.update_score(6.0, AssessmentSource.SINGLE_QA)
        score.update_score(7.0, AssessmentSource.MOCK_INTERVIEW)
        score.update_score(7.5, AssessmentSource.DIAGNOSTIC)

        assert len(score.score_history) == 3
        assert score.assessment_count == 3

    def test_trend_detection_improving(self):
        """Detects improving trend when scores increase consistently."""
        score = SkillScore(
            topic="system_design",
            category=SkillCategory.SYSTEM_DESIGN,
            score=4.0,
        )
        score.update_score(5.0, AssessmentSource.SINGLE_QA)
        score.update_score(6.0, AssessmentSource.SINGLE_QA)
        score.update_score(7.0, AssessmentSource.SINGLE_QA)

        assert score.trend == "improving"

    def test_source_affects_weight(self):
        """Diagnostic assessment has more impact than self-reported.

        A DIAGNOSTIC (structured test) should shift the score more than
        a SELF_REPORTED claim ("I think I'm good at this").
        """
        score1 = SkillScore(topic="a", category=SkillCategory.MACHINE_LEARNING, score=5.0, confidence=0.3)
        score2 = SkillScore(topic="b", category=SkillCategory.MACHINE_LEARNING, score=5.0, confidence=0.3)

        score1.update_score(8.0, AssessmentSource.DIAGNOSTIC)
        score2.update_score(8.0, AssessmentSource.SELF_REPORTED)

        # Diagnostic should shift more than self-reported
        assert score1.score > score2.score, "Diagnostic should have more impact than self-report"

    def test_score_validation_bounds(self):
        """Score stays within 0-10 range."""
        with pytest.raises(Exception):
            SkillScore(topic="x", category=SkillCategory.MACHINE_LEARNING, score=11.0)
        with pytest.raises(Exception):
            SkillScore(topic="x", category=SkillCategory.MACHINE_LEARNING, score=-1.0)


# ════════════════════════════════════════════════════════════
# 2. USER PROFILE TESTS
# ════════════════════════════════════════════════════════════

class TestUserProfile:
    """Test profile management and priority calculations."""

    def test_profile_creation_defaults(self):
        """Profile creates with sensible defaults."""
        profile = UserProfile(name="Prudhvi", target_role="Senior ML Engineer")
        assert profile.name == "Prudhvi"
        assert profile.skill_scores == {}
        assert profile.profile_version == 0

    def test_skill_update_creates_new(self):
        """First update on a topic creates the SkillScore entry."""
        profile = UserProfile(name="Test")
        profile.update_skill(
            "transformers", 7.0,
            AssessmentSource.DIAGNOSTIC,
            SkillCategory.DEEP_LEARNING,
        )
        assert "transformers" in profile.skill_scores
        assert profile.skill_scores["transformers"].score == 7.0
        assert profile.profile_version == 1

    def test_skill_update_bayesian_on_existing(self):
        """Subsequent updates use Bayesian blending, not overwrite."""
        profile = UserProfile(name="Test")
        profile.update_skill("sql", 5.0, AssessmentSource.DIAGNOSTIC, SkillCategory.SQL_CODING)
        profile.update_skill("sql", 8.0, AssessmentSource.MOCK_INTERVIEW)

        # Should be between 5 and 8, not exactly 8
        assert 5.0 < profile.skill_scores["sql"].score < 8.0

    def test_priority_topics(self):
        """Priority calculation returns weakest topics first."""
        profile = UserProfile(name="Test")
        profile.update_skill("strong_topic", 9.0, AssessmentSource.DIAGNOSTIC, SkillCategory.MACHINE_LEARNING)
        profile.update_skill("weak_topic", 3.0, AssessmentSource.DIAGNOSTIC, SkillCategory.SYSTEM_DESIGN)
        profile.update_skill("medium_topic", 6.0, AssessmentSource.DIAGNOSTIC, SkillCategory.STATISTICS)

        priorities = profile.get_priority_topics(3)
        assert priorities[0]["topic"] == "weak_topic"  # Weakest = highest priority

    def test_readiness_summary(self):
        """Readiness summary includes all expected fields."""
        profile = UserProfile(name="Test")
        profile.update_skill("a", 8.0, AssessmentSource.DIAGNOSTIC, SkillCategory.DEEP_LEARNING)
        profile.update_skill("b", 3.0, AssessmentSource.DIAGNOSTIC, SkillCategory.STATISTICS)

        summary = profile.get_readiness_summary()
        assert "average_score" in summary
        assert "weakest_topics" in summary
        assert "strongest_topics" in summary


# ════════════════════════════════════════════════════════════
# 3. MEMORY TIER TESTS
# ════════════════════════════════════════════════════════════

class TestConversationBuffer:
    """Test Tier 1: short-term conversation buffer."""

    def test_add_and_retrieve(self):
        buf = ConversationBuffer(max_messages=5)
        buf.add_message("user", "hello")
        buf.add_message("assistant", "hi there")
        assert len(buf.get_messages()) == 2

    def test_sliding_window_eviction(self):
        """Oldest messages evicted when buffer is full."""
        buf = ConversationBuffer(max_messages=3)
        buf.add_message("user", "msg1")
        buf.add_message("assistant", "msg2")
        buf.add_message("user", "msg3")
        buf.add_message("assistant", "msg4")  # This should evict msg1

        messages = buf.get_messages()
        assert len(messages) == 3
        assert messages[0]["content"] == "msg2"  # msg1 evicted

    def test_to_prompt_messages_strips_timestamps(self):
        """Prompt messages don't include metadata — just role + content."""
        buf = ConversationBuffer()
        buf.add_message("user", "test")
        prompt_msgs = buf.to_prompt_messages()
        assert "timestamp" not in prompt_msgs[0]
        assert prompt_msgs[0]["role"] == "user"


class TestSessionSummaryStore:
    """Test Tier 2: session summaries."""

    def test_summarize_session(self):
        mock_llm = MockLLMGateway("User practiced transformers. Scored 7/10.")
        store = SessionSummaryStore(llm=mock_llm)

        messages = [
            {"role": "user", "content": "Ask me about transformers"},
            {"role": "assistant", "content": "Explain self-attention."},
            {"role": "user", "content": "It computes Q, K, V..."},
        ]
        summary = store.summarize_session(messages)

        assert summary == "User practiced transformers. Scored 7/10."
        assert store.get_all_summaries_count() == 1

    def test_get_recent_summaries(self):
        mock_llm = MockLLMGateway("Summary")
        store = SessionSummaryStore(llm=mock_llm)

        for i in range(5):
            store.summarize_session([{"role": "user", "content": f"session {i}"}])

        recent = store.get_recent_summaries(k=3)
        assert len(recent) == 3


class TestSemanticMemory:
    """Test Tier 3: long-term vector memory."""

    def test_store_and_retrieve(self):
        mock_llm = MockLLMGateway()
        mem = SemanticMemory(llm=mock_llm)

        mem.store_fact("User is weak at system design")
        mem.store_fact("User scored 8/10 on transformers")
        mem.store_fact("User prefers practical examples")

        results = mem.retrieve("system design weakness", k=2)
        assert len(results) > 0
        assert any("system design" in r.lower() for r in results)

    def test_empty_retrieval(self):
        mock_llm = MockLLMGateway()
        mem = SemanticMemory(llm=mock_llm)
        results = mem.retrieve("anything")
        assert results == []


class TestMemoryManager:
    """Test unified memory manager."""

    def test_full_context_assembly(self):
        mock_llm = MockLLMGateway("Summary of past session")
        manager = MemoryManager(llm=mock_llm)

        # Add to all tiers
        manager.add_message("user", "hello")
        manager.store_fact("User targets Google")

        # End session to create summary
        manager.add_message("user", "let's practice transformers")
        manager.add_message("assistant", "Explain self-attention")
        manager.end_session()

        # Get unified context
        context = manager.get_context(query="Google preparation")
        assert isinstance(context, str)

    def test_memory_stats(self):
        mock_llm = MockLLMGateway()
        manager = MemoryManager(llm=mock_llm)
        manager.add_message("user", "test")
        manager.store_fact("test fact")

        stats = manager.get_memory_stats()
        assert stats["buffer_messages"] == 1
        assert stats["long_term_facts"] == 1


# ════════════════════════════════════════════════════════════
# 4. TOKEN BUDGET TESTS
# ════════════════════════════════════════════════════════════

class TestTokenBudgetAllocator:
    """Test token budget allocation and truncation."""

    def test_allocation_report(self):
        allocator = TokenBudgetAllocator()
        report = allocator.allocate({
            "system_prompt": "You are a helpful assistant.",
            "current_input": "Hello world",
        })

        assert "allocations" in report
        assert "total_input_tokens" in report
        assert "remaining_capacity" in report

    def test_over_budget_warning(self):
        """Components exceeding their budget generate warnings."""
        allocator = TokenBudgetAllocator()
        allocator.set_budget("current_input", 5)  # Very small budget

        report = allocator.allocate({
            "current_input": "This is a long message that definitely exceeds five tokens",
        })

        assert len(report["warnings"]) > 0
        assert report["allocations"]["current_input"]["over_budget"] is True

    def test_truncation_respects_budget(self):
        allocator = TokenBudgetAllocator()
        allocator.set_budget("test_component", 10)

        long_text = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\nLine 6\nLine 7\nLine 8"
        truncated = allocator.truncate_to_budget(long_text, "test_component")

        tokens = allocator.count_tokens(truncated)
        assert tokens <= 10

    def test_budget_summary(self):
        allocator = TokenBudgetAllocator()
        summary = allocator.get_budget_summary()
        assert "budgets" in summary
        assert "model_limit" in summary


# ════════════════════════════════════════════════════════════
# 5. PERSISTENCE TESTS (SQLite)
# ════════════════════════════════════════════════════════════

class TestProfileStore:
    """Test SQLite persistence with temporary database."""

    @pytest.fixture
    def store(self):
        """Create a temporary database for each test."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        store = ProfileStore(db_path=db_path)
        yield store
        store.close()  # Release file lock before cleanup
        try:
            os.unlink(db_path)
        except PermissionError:
            pass  # Windows may still hold lock briefly; temp dir cleans up later

    def test_save_and_load(self, store):
        """Profile survives save → load roundtrip."""
        profile = UserProfile(name="Prudhvi", target_role="Senior ML Engineer")
        profile.update_skill("transformers", 7.5, AssessmentSource.DIAGNOSTIC, SkillCategory.DEEP_LEARNING)

        store.save_profile(profile)
        loaded = store.load_profile(profile.user_id)

        assert loaded is not None
        assert loaded.name == "Prudhvi"
        assert "transformers" in loaded.skill_scores
        assert loaded.skill_scores["transformers"].score == 7.5

    def test_upsert_updates_existing(self, store):
        """Saving an existing profile updates it, doesn't create duplicate."""
        profile = UserProfile(name="Test")
        store.save_profile(profile)

        profile.name = "Updated"
        profile.profile_version += 1
        store.save_profile(profile)

        loaded = store.load_profile(profile.user_id)
        assert loaded.name == "Updated"

        all_profiles = store.list_profiles()
        assert len(all_profiles) == 1  # No duplicates

    def test_load_nonexistent_returns_none(self, store):
        """Loading non-existent profile returns None, not error."""
        result = store.load_profile("nonexistent_id")
        assert result is None

    def test_delete_profile(self, store):
        profile = UserProfile(name="Delete Me")
        store.save_profile(profile)
        assert store.delete_profile(profile.user_id) is True
        assert store.load_profile(profile.user_id) is None

    def test_session_summary_persistence(self, store):
        store.save_summary("user_1", "Practiced transformers, scored 7/10", 15)
        summaries = store.get_summaries("user_1")
        assert len(summaries) == 1
        assert "transformers" in summaries[0]["summary"]

    def test_memory_fact_persistence(self, store):
        store.save_fact("user_1", "User is weak at system design")
        facts = store.get_facts("user_1")
        assert len(facts) == 1
        assert "system design" in facts[0]["fact"]


# ════════════════════════════════════════════════════════════
# 6. INTERVIEW SESSION TESTS
# ════════════════════════════════════════════════════════════

class TestInterviewSession:
    """Test mock interview session management."""

    def test_session_lifecycle(self):
        session = InterviewSession(
            session_id="test_001",
            topic="transformers",
            difficulty=3,
            is_active=True,
        )
        session.started_at = session.started_at or __import__("datetime").datetime.now()

        # Add question and answer
        session.add_question({"question": "Explain self-attention", "difficulty": 3})
        session.add_answer({
            "answer": "Self-attention computes Q, K, V...",
            "scores": {"clarity": 7, "depth": 6, "structure": 7, "relevance": 8, "communication": 7},
        })

        assert session.current_question_index == 1
        assert session.average_score == 7.0

        # End session
        summary = session.end_session()
        assert session.is_active is False
        assert summary["average_score"] == 7.0