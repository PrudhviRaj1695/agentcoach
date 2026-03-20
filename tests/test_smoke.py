"""Smoke test — verifies project structure is correct."""

def test_imports():
    """Verify all packages are importable."""
    from src.models.state import AgentState, UserProfile, SkillScore
    from src.config.settings import Settings
    
    # State schema creates without error
    state: AgentState = {
        "messages": [],
        "current_input": "hello",
        "current_mode": "chat",
        "step_count": 0,
        "is_complete": False,
    }
    assert state["current_mode"] == "chat"
    
    # Pydantic models validate
    profile = UserProfile(name="Test", target_role="Senior ML Engineer")
    assert profile.name == "Test"
    
    score = SkillScore(topic="transformers", score=7.5)
    assert score.score == 7.5


def test_settings_defaults():
    """Settings load with defaults (no .env needed for CI)."""
    from src.config.settings import Settings
    s = Settings()
    assert s.app_env == "development"