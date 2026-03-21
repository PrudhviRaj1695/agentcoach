"""Enhanced state models for AgentCoach — Story 2.1.

═══════════════════════════════════════════════════════════════════
WHY THIS FILE EXISTS (Interview Context):
═══════════════════════════════════════════════════════════════════

Interviewers will ask: "How do you model user state in your agentic system?"

The WRONG answer: "I just store everything in a dict."
The RIGHT answer: "Typed Pydantic models with validation, versioning,
and clear separation between transient state (conversation) and persistent
state (user profile, skill scores)."

This file defines THREE layers of data:
1. UserProfile — persistent across sessions (WHO the user is)
2. SkillScore — tracked over time with Bayesian updates (HOW they're improving)
3. InterviewSession — transient per mock interview (WHAT they're doing now)
4. AgentState — LangGraph's shared state (TypedDict for graph compatibility)

Key interview phrase: "I separate persistent user data from transient
conversation state. The UserProfile persists in SQLite across sessions,
while AgentState is recreated per conversation turn. This separation lets
me checkpoint conversations without corrupting the user's long-term profile."

═══════════════════════════════════════════════════════════════════
COMPARISON WITH PRE-AUTH (Interviewers will ask):
═══════════════════════════════════════════════════════════════════

Pre-Auth State:
  - claim_id, patient_context, retrieved_policies, validation_result
  - Lifetime: single claim processing (seconds)
  - No cross-claim memory

AgentCoach State:
  - user_profile with skill_scores, interview_dates, target_companies
  - Lifetime: weeks/months (full preparation journey)
  - Cross-session memory with improvement tracking

This is the difference between a TRANSACTION system and a RELATIONSHIP system.
"""

from datetime import datetime
from enum import Enum
from typing import TypedDict

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────
# ENUMS — Interview talking point: "Explicit enums, not magic strings"
# ─────────────────────────────────────────────────────────────

class SkillCategory(str, Enum):
    """Top-level skill categories we assess.

    WHY ENUM not strings: prevents typos like "machin_learning" vs
    "machine_learning". Pydantic validates on assignment. If an LLM
    returns a category not in this enum, it fails validation and we
    catch it immediately rather than silently storing garbage.

    Interview phrase: "Enum-based categories with Pydantic validation
    catch LLM output errors at the boundary — before they corrupt state."
    """
    MACHINE_LEARNING = "machine_learning"
    DEEP_LEARNING = "deep_learning"
    NLP_LLMS = "nlp_llms"
    STATISTICS = "statistics"
    SQL_CODING = "sql_coding"
    SYSTEM_DESIGN = "system_design"
    BEHAVIORAL = "behavioral"
    MLOPS = "mlops"


class AssessmentSource(str, Enum):
    """How was this skill score generated?

    WHY THIS MATTERS: In interviews, they'll ask "how do you know
    the skill score is reliable?" This enum tracks provenance.
    A score from a 10-question mock interview is more reliable
    than a score from one casual answer.

    Interview phrase: "Every skill score has provenance — I track whether
    it came from a diagnostic, a mock interview, or a single Q&A.
    This feeds into confidence weighting — a 10-question assessment
    is weighted higher than a single casual answer."
    """
    DIAGNOSTIC = "diagnostic"           # Initial skill assessment (3-5 questions)
    MOCK_INTERVIEW = "mock_interview"   # Full mock interview session
    SINGLE_QA = "single_qa"             # One-off question during conversation
    SELF_REPORTED = "self_reported"      # User told us their level
    INFERRED = "inferred"               # Inferred from related skills


# ─────────────────────────────────────────────────────────────
# SKILL TRACKING — The core of adaptive coaching
# ─────────────────────────────────────────────────────────────

class SkillScore(BaseModel):
    """Individual skill assessment with Bayesian-style tracking.

    ═══════════════════════════════════════════════════════════════
    WHY BAYESIAN UPDATES (Interview Deep Dive):
    ═══════════════════════════════════════════════════════════════

    Naive approach: overwrite score each time.
    Problem: User scores 8/10 on transformers, then scores 4/10 on
    one hard question. Naive update drops them to 4. That's wrong —
    one bad answer shouldn't erase demonstrated competence.

    Bayesian approach: treat each assessment as EVIDENCE that updates
    a belief distribution. More assessments = higher confidence = less
    volatile. The formula:

        new_score = (old_score * old_confidence + new_evidence * evidence_weight)
                    / (old_confidence + evidence_weight)

    This is essentially a weighted moving average where the weight
    depends on how many assessments we've done.

    Interview phrase: "Bayesian skill updates — each interaction
    refines the estimate rather than replacing it. A user with 20
    assessments on transformers won't have their score destroyed by
    one bad answer, because the confidence is high. But a new topic
    with only 1 assessment will shift significantly on new evidence."

    ═══════════════════════════════════════════════════════════════
    WHY THESE SPECIFIC FIELDS (each one has a reason):
    ═══════════════════════════════════════════════════════════════
    """

    topic: str
    """What skill this tracks. Maps to a specific topic in our curriculum.
    Example: "transformers", "gradient_boosting", "sql_window_functions" """

    category: SkillCategory
    """Which broad area. Used for study plan balancing — we ensure
    the user doesn't spend 100% of time on ML and 0% on system design."""

    score: float = Field(ge=0, le=10, default=5.0)
    """Current skill estimate (0-10 scale).
    - 0-3: No knowledge (can't explain the concept)
    - 4-6: Conceptual (can explain but not apply)
    - 7-8: Applied (can implement and discuss trade-offs)
    - 9-10: Expert (can teach, debug, and optimize)

    WHY 0-10 not 0-100: Finer granularity creates false precision.
    The difference between 73 and 75 is meaningless for interview prep.
    0-10 maps cleanly to verbal labels that users understand."""

    confidence: float = Field(ge=0, le=1, default=0.1)
    """How confident are we in this score? (0-1)
    Starts at 0.1 (we know nothing). Increases with each assessment.

    WHY THIS MATTERS: Low confidence = volatile score (will shift a lot
    on new evidence). High confidence = stable score (needs strong
    contrary evidence to shift).

    Formula: confidence = min(1.0, assessment_count * 0.1 + base_confidence)
    After 10 assessments, confidence = 1.0 (fully confident)."""

    assessment_count: int = 0
    """How many times we've assessed this skill.
    Directly feeds confidence calculation.
    Also used for: "You've practiced transformers 15 times — ready to move on." """

    last_assessed: datetime = Field(default_factory=datetime.now)
    """When was this last assessed?
    WHY: Skills decay. A score from 3 weeks ago is less reliable than
    yesterday's score. We apply time decay in priority calculations:
    priority = (10 - score) * recency_weight * company_relevance """

    trend: str = Field(default="stable")
    """improving / stable / declining — computed from score history.
    WHY: A user stuck at 5/10 for 3 weeks on a topic needs a different
    intervention than a user who was at 3/10 last week and is now at 5/10.
    Stagnation detection triggers strategy changes."""

    score_history: list[dict] = Field(default_factory=list)
    """List of {score, timestamp, source} for trend analysis.
    WHY: Without history, we can't show improvement trajectories.
    The interview phrase: "I track the full score history per topic
    so I can detect stagnation — a topic that hasn't improved in 3
    sessions despite practice needs a different approach." """

    source: AssessmentSource = AssessmentSource.INFERRED
    """How was the latest score generated? Provenance tracking."""

    def update_score(self, new_evidence: float, source: AssessmentSource) -> None:
        """Bayesian-style score update.

        This is the KEY algorithm. Rather than overwriting, we blend
        the new evidence with the existing belief, weighted by confidence.

        Example walkthrough:
        - User has score=7.0, confidence=0.5 (5 assessments) on "transformers"
        - They score 4.0 on a hard question (evidence_weight based on source)
        - new_score = (7.0 * 0.5 + 4.0 * 0.2) / (0.5 + 0.2) = 5.86
        - Score drops from 7.0 to 5.86, not to 4.0

        If same user had confidence=0.9 (9 assessments):
        - new_score = (7.0 * 0.9 + 4.0 * 0.2) / (0.9 + 0.2) = 6.45
        - Much less drop! High confidence = high stability.

        Interview phrase: "The update formula is a confidence-weighted
        Bayesian blend. High-confidence topics are stable, low-confidence
        topics are volatile. This prevents one bad answer from destroying
        a well-established skill estimate."
        """
        # Evidence weight depends on source quality
        evidence_weights = {
            AssessmentSource.DIAGNOSTIC: 0.3,       # Structured assessment = strong signal
            AssessmentSource.MOCK_INTERVIEW: 0.25,   # Full interview = strong signal
            AssessmentSource.SINGLE_QA: 0.15,        # One question = weaker signal
            AssessmentSource.SELF_REPORTED: 0.1,     # User claims = weakest signal
            AssessmentSource.INFERRED: 0.05,         # Indirect = minimal signal
        }
        evidence_weight = evidence_weights.get(source, 0.1)

        # Bayesian blend
        old_weight = self.confidence
        new_weight = evidence_weight
        self.score = round(
            (self.score * old_weight + new_evidence * new_weight)
            / (old_weight + new_weight),
            2,
        )

        # Update metadata
        self.assessment_count += 1
        self.confidence = min(1.0, self.assessment_count * 0.1 + 0.1)
        self.last_assessed = datetime.now()
        self.source = source

        # Append to history for trend analysis
        self.score_history.append({
            "score": self.score,
            "raw_evidence": new_evidence,
            "source": source.value,
            "timestamp": self.last_assessed.isoformat(),
        })

        # Update trend (needs at least 3 data points)
        self._update_trend()

    def _update_trend(self) -> None:
        """Detect if skill is improving, stable, or declining.

        Uses last 5 scores. If average of recent 3 > average of older 2,
        it's improving. This simple heuristic works well enough — we don't
        need linear regression for 5 data points.

        Interview phrase: "Trend detection using rolling window comparison.
        If a topic is declining despite practice, the system flags it
        for strategy change — maybe the user needs a different learning
        resource, not just more repetition."
        """
        if len(self.score_history) < 3:
            self.trend = "stable"
            return

        recent = self.score_history[-3:]
        older = self.score_history[-5:-3] if len(self.score_history) >= 5 else self.score_history[:2]

        avg_recent = sum(h["score"] for h in recent) / len(recent)
        avg_older = sum(h["score"] for h in older) / len(older) if older else avg_recent

        diff = avg_recent - avg_older
        if diff > 0.2:
            self.trend = "improving"
        elif diff < -0.2:
            self.trend = "declining"
        else:
            self.trend = "stable"


# ─────────────────────────────────────────────────────────────
# USER PROFILE — The persistent identity
# ─────────────────────────────────────────────────────────────

class UserProfile(BaseModel):
    """Persistent user profile — survives across all sessions.

    ═══════════════════════════════════════════════════════════════
    WHY EACH FIELD EXISTS:
    ═══════════════════════════════════════════════════════════════

    Every field serves a SPECIFIC purpose in the coaching system.
    No field is decorative. If it doesn't affect routing, planning,
    or evaluation, it doesn't belong here.

    Interview phrase: "Every profile field feeds a downstream decision.
    target_companies drives study plan prioritization. experience_years
    calibrates question difficulty. interview_dates drive urgency
    calculations. No decorative fields."
    """

    # Identity
    user_id: str = Field(default_factory=lambda: f"user_{datetime.now().strftime('%Y%m%d%H%M%S')}")
    name: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # Career context — drives study plan content
    target_role: str = ""
    """e.g., "Senior ML Engineer", "Staff Data Scientist"
    WHY: Determines which topics are essential vs nice-to-have.
    A Staff MLE needs system design depth. A Senior DS needs stats depth."""

    target_companies: list[str] = Field(default_factory=list)
    """e.g., ["Google", "Amazon", "Meta"]
    WHY: Each company has different interview patterns.
    Google = theory-heavy. Amazon = applied + leadership principles.
    This drives company-specific strategy (Epic 3, Story 3.3)."""

    experience_years: float = 0
    """Years of relevant experience.
    WHY: Calibrates question difficulty and expected depth.
    <3 years = basic conceptual. 3-7 = applied + trade-offs.
    7+ = system design + architecture + mentoring stories."""

    tech_stack: list[str] = Field(default_factory=list)
    """e.g., ["Python", "PyTorch", "LangChain", "Azure"]
    WHY: When generating mock questions, we tailor to their stack.
    No point asking about TensorFlow Serving if they use Azure ML."""

    # Timeline — drives urgency
    interview_dates: dict[str, str] = Field(default_factory=dict)
    """{"Google": "2026-04-15", "Amazon": "2026-05-01"}
    WHY: The #1 input to study plan prioritization.
    If Google is in 2 weeks, focus on Google patterns.
    If Amazon is in 6 weeks, there's time for broader prep."""

    # Skill tracking — the intelligence layer
    skill_scores: dict[str, SkillScore] = Field(default_factory=dict)
    """Maps topic_name → SkillScore.
    WHY: This IS the adaptive intelligence. Every coaching decision
    flows from skill scores: what to study, what to practice,
    what difficulty to set in mock interviews."""

    # Preferences
    preferred_study_hours_per_day: float = Field(ge=0, le=12, default=2.0)
    """How many hours per day can the user study?
    WHY: A plan that assumes 8 hours/day for someone who has 2 hours
    is useless. This constraint-satisfies the study plan."""

    weak_areas_self_reported: list[str] = Field(default_factory=list)
    """What the user THINKS they're weak at.
    WHY: Often diverges from actual skill scores. The gap between
    self-perception and measured ability is itself a coaching signal.
    "You think you're weak at DL, but you scored 7.5 — your actual
    gap is system design where you scored 4.0." """

    # Version tracking
    profile_version: int = 0
    """Incremented on every update.
    WHY: Audit trail. If something goes wrong with the profile,
    we can trace when it changed. Also prevents stale-data bugs
    in concurrent access scenarios."""

    def get_priority_topics(self, top_k: int = 5) -> list[dict]:
        """Get highest-priority topics to study based on skill gaps.

        Priority formula:
            priority = (10 - score) * recency_factor * urgency_factor

        Where:
        - (10 - score): bigger gap = higher priority
        - recency_factor: stale assessments get boosted (might have improved)
        - urgency_factor: nearer interview = higher priority

        Interview phrase: "Priority scoring combines skill gap magnitude,
        assessment recency, and interview urgency. A low score on a topic
        relevant to your nearest interview gets the highest priority."
        """
        priorities = []
        for topic, skill in self.skill_scores.items():
            gap = 10 - skill.score

            # Recency factor: older assessments get priority boost
            days_since = (datetime.now() - skill.last_assessed).days
            recency_factor = min(2.0, 1.0 + days_since * 0.05)

            priority = gap * recency_factor

            priorities.append({
                "topic": topic,
                "category": skill.category.value,
                "score": skill.score,
                "confidence": skill.confidence,
                "trend": skill.trend,
                "priority": round(priority, 2),
                "assessment_count": skill.assessment_count,
            })

        priorities.sort(key=lambda x: x["priority"], reverse=True)
        return priorities[:top_k]

    def update_skill(
        self, topic: str, score: float, source: AssessmentSource,
        category: SkillCategory | None = None,
    ) -> None:
        """Update a skill score with Bayesian blending.

        If the topic doesn't exist yet, creates it.
        If it exists, applies Bayesian update.
        """
        if topic in self.skill_scores:
            self.skill_scores[topic].update_score(score, source)
        else:
            self.skill_scores[topic] = SkillScore(
                topic=topic,
                category=category or SkillCategory.MACHINE_LEARNING,
                score=score,
                confidence=0.15,
                assessment_count=1,
                source=source,
                score_history=[{
                    "score": score,
                    "raw_evidence": score,
                    "source": source.value,
                    "timestamp": datetime.now().isoformat(),
                }],
            )

        self.updated_at = datetime.now()
        self.profile_version += 1

    def get_readiness_summary(self) -> dict:
        """Overall readiness assessment.

        Interview phrase: "The readiness summary gives a snapshot of
        where the user stands — average score, weakest areas, strongest
        areas, and topics that haven't been assessed yet. This feeds
        the study planner and the mock interviewer's difficulty selection."
        """
        if not self.skill_scores:
            return {"status": "no_assessments", "message": "No skills assessed yet."}

        scores = [s.score for s in self.skill_scores.values()]
        return {
            "average_score": round(sum(scores) / len(scores), 1),
            "total_topics_assessed": len(scores),
            "weakest_topics": self.get_priority_topics(3),
            "strongest_topics": sorted(
                [{"topic": t, "score": s.score} for t, s in self.skill_scores.items()],
                key=lambda x: x["score"],
                reverse=True,
            )[:3],
            "declining_topics": [
                t for t, s in self.skill_scores.items() if s.trend == "declining"
            ],
            "stagnant_topics": [
                t for t, s in self.skill_scores.items()
                if s.trend == "stable" and s.assessment_count > 3 and s.score < 7
            ],
        }


# ─────────────────────────────────────────────────────────────
# INTERVIEW SESSION — Transient per mock interview
# ─────────────────────────────────────────────────────────────

class InterviewSession(BaseModel):
    """Active mock interview session state.

    WHY SEPARATE FROM PROFILE: A mock interview is transient — it starts
    and ends within a conversation. The Profile persists forever. Mixing
    them would mean every interview's temporary data pollutes the
    permanent profile.

    Interview phrase: "I separate transient session state from persistent
    profile state. The interview session tracks questions, answers, and
    scores for the current mock interview. When the session ends, the
    SCORES flow back to the profile via Bayesian update, but the session
    itself is archived — not kept in the hot profile."
    """

    session_id: str = ""
    session_type: str = "mock_interview"  # mock_interview | diagnostic | quick_practice

    # Configuration
    topic: str = ""
    target_company: str = ""
    difficulty: int = Field(ge=1, le=5, default=3)
    persona: str = "neutral"  # friendly | neutral | aggressive
    max_questions: int = 5

    # Progress
    questions_asked: list[dict] = Field(default_factory=list)
    """[{question, topic, difficulty, expected_answer_outline, timestamp}]"""

    answers_given: list[dict] = Field(default_factory=list)
    """[{answer, scores: {clarity, depth, structure, relevance, communication}, feedback}]"""

    current_question_index: int = 0
    is_active: bool = False
    started_at: datetime | None = None
    ended_at: datetime | None = None

    # Aggregate metrics
    average_score: float = 0.0
    topics_covered: list[str] = Field(default_factory=list)

    def add_question(self, question: dict) -> None:
        """Record a question asked during the session."""
        question["timestamp"] = datetime.now().isoformat()
        self.questions_asked.append(question)
        self.current_question_index = len(self.questions_asked)

    def add_answer(self, answer: dict) -> None:
        """Record an answer and update running average."""
        self.answers_given.append(answer)
        if "scores" in answer:
            all_scores = []
            for a in self.answers_given:
                if "scores" in a:
                    # Average across dimensions for this answer
                    dims = a["scores"]
                    avg = sum(dims.values()) / len(dims) if dims else 0
                    all_scores.append(avg)
            self.average_score = round(sum(all_scores) / len(all_scores), 2) if all_scores else 0

    def end_session(self) -> dict:
        """End session and return summary for profile update.

        Returns the data needed to update the user's profile skill scores.
        """
        self.is_active = False
        self.ended_at = datetime.now()

        return {
            "session_id": self.session_id,
            "topic": self.topic,
            "questions_count": len(self.questions_asked),
            "average_score": self.average_score,
            "topics_covered": list(set(self.topics_covered)),
            "duration_minutes": (
                (self.ended_at - self.started_at).total_seconds() / 60
                if self.started_at else 0
            ),
        }


# ─────────────────────────────────────────────────────────────
# LANGGRAPH STATE — Updated with memory fields
# ─────────────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    """Shared state passed between all agents in LangGraph.

    ═══════════════════════════════════════════════════════════════
    WHY TypedDict (not Pydantic) for graph state:
    ═══════════════════════════════════════════════════════════════

    LangGraph requires TypedDict for state because it needs to:
    1. Merge partial state updates from agents (TypedDict supports this)
    2. Track which fields changed (for checkpointing)
    3. Serialize/deserialize for persistence

    Pydantic models are used INSIDE the state (UserProfile, SkillScore)
    but serialized to dicts when stored in the TypedDict. This gives us
    validation at the model level + compatibility at the graph level.

    Interview phrase: "TypedDict for LangGraph compatibility, Pydantic
    models for validation. Agents receive TypedDict state, deserialize
    relevant fields into Pydantic models for type-safe operations, then
    serialize back to TypedDict for the next agent."
    ═══════════════════════════════════════════════════════════════
    """

    # User context (persistent)
    user_profile: dict              # Serialized UserProfile

    # Current conversation (transient)
    messages: list[dict]            # Chat history [{role, content}]
    current_input: str              # Latest user message

    # Routing
    current_mode: str               # planning|interviewing|evaluating|strategy|chat|profile
    current_agent: str              # Active agent name
    route_confidence: float         # Router confidence (0-1)

    # Agent output
    agent_response: str             # Response to return to user

    # Interview session (transient)
    active_interview: dict          # Serialized InterviewSession

    # Study plan (semi-persistent)
    study_plan: dict                # Current active plan

    # ══ NEW IN EPIC 2: Memory fields ══
    memory_context: str             # Retrieved long-term memories
    session_summary: str            # Summary of current session
    token_budget: dict              # Token allocation per component

    # Control
    step_count: int                 # Loop guard (max 5)
    is_complete: bool               # Terminal flag
    error: str | None               # Error capture