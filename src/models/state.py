"""Core state schema for LangGraph — the backbone of the system."""

from datetime import datetime
from typing import TypedDict

from pydantic import BaseModel, Field


# ── Pydantic models for validation ──

class SkillScore(BaseModel):
    """Individual skill assessment."""
    topic: str
    score: float = Field(ge=0, le=10)
    confidence: float = Field(ge=0, le=1, default=0.5)
    last_assessed: datetime = Field(default_factory=datetime.now)
    assessment_count: int = 0


class UserProfile(BaseModel):
    """Persistent user profile."""
    name: str = ""
    target_role: str = ""
    target_companies: list[str] = Field(default_factory=list)
    experience_years: float = 0
    tech_stack: list[str] = Field(default_factory=list)
    interview_dates: dict[str, str] = Field(default_factory=dict)
    skill_scores: dict[str, SkillScore] = Field(default_factory=dict)


class InterviewSession(BaseModel):
    """Active interview session state."""
    session_id: str = ""
    topic: str = ""
    difficulty: int = Field(ge=1, le=5, default=3)
    questions_asked: list[dict] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)
    is_active: bool = False


# ── LangGraph State (TypedDict for graph compatibility) ──

class AgentState(TypedDict, total=False):
    """Shared state passed between all agents in LangGraph."""
    
    # User context
    user_profile: dict          # Serialized UserProfile
    
    # Current conversation
    messages: list[dict]        # Chat history [{role, content}]
    current_input: str          # Latest user message
    
    # Routing
    current_mode: str           # planning | interviewing | evaluating | strategy | chat
    current_agent: str          # Which agent is active
    route_confidence: float     # Router's confidence in classification
    
    # Agent outputs
    agent_response: str         # Current agent's response
    
    # Interview session
    active_interview: dict      # Serialized InterviewSession
    
    # Study plan
    study_plan: dict            # Current active plan
    
    # Memory
    memory_context: str         # Retrieved long-term memory
    session_summary: str        # Summary of current session
    
    # Control
    step_count: int             # Guard against infinite loops
    is_complete: bool           # Terminal condition
    error: str | None           # Error message if any