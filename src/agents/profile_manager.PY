"""ProfileManager Agent — Story 2.1.

═══════════════════════════════════════════════════════════════════
WHY THIS AGENT EXISTS (Interview Context):
═══════════════════════════════════════════════════════════════════

Problem: Users don't fill out forms. They share information naturally
in conversation: "I have 5 years of ML experience, targeting Google
and Amazon, my interview is in 3 weeks."

This agent EXTRACTS structured profile data from unstructured conversation.

This is an NLP EXTRACTION task — the same pattern used in:
- Named Entity Recognition (NER)
- Information Extraction from clinical notes (your healthcare experience)
- Slot filling in conversational AI

Interview phrase: "The ProfileManager is an information extraction agent.
It uses GPT-4o with structured output to extract profile fields from
natural conversation — the same pattern as clinical NER but applied to
career context. I chose LLM-based extraction over regex because user
messages are too varied for rule-based parsing."

═══════════════════════════════════════════════════════════════════
ARCHITECTURAL DECISION: LLM extraction vs form-based input
═══════════════════════════════════════════════════════════════════

Option A: Ask users to fill a structured form
  Pro: Clean data, no extraction errors
  Con: Terrible UX, users drop off, feels like a survey

Option B: LLM extracts from conversation (our choice)
  Pro: Natural UX, users share more information freely
  Con: Extraction can miss fields, need validation

Option C: Hybrid — extract from conversation + confirm with user
  Pro: Best of both worlds (our actual implementation)
  Con: Slightly more complex flow

We chose Option C. The agent extracts what it can, then asks targeted
follow-up questions for missing critical fields.

Interview phrase: "Hybrid extraction — the LLM extracts what it can
from natural conversation, then asks targeted follow-ups for missing
fields. This avoids the 'form filling' UX while ensuring we get the
data we need for personalization."
"""

import json
from datetime import datetime

from src.agents.base import BaseAgent
from src.config.llm_gateway import LLMGateway, ModelTier
from src.models.state import (
    AgentState,
    AssessmentSource,
    SkillCategory,
    UserProfile,
)


# Fields we MUST have before we can start coaching effectively
REQUIRED_FIELDS = ["name", "target_role", "experience_years"]
# Fields that improve coaching quality but aren't blocking
OPTIONAL_FIELDS = ["target_companies", "tech_stack", "interview_dates"]


PROFILE_EXTRACTION_PROMPT = """You are a profile extraction system for an AI interview coach.

Given the conversation, extract any user profile information mentioned.

Return ONLY valid JSON with these fields (include ONLY fields explicitly mentioned):
{
    "name": "string or null",
    "target_role": "string or null",
    "target_companies": ["list of company names"] or null,
    "experience_years": number or null,
    "tech_stack": ["list of technologies"] or null,
    "interview_dates": {"company_name": "YYYY-MM-DD"} or null,
    "preferred_study_hours_per_day": number or null,
    "weak_areas_self_reported": ["list of topic areas"] or null
}

Rules:
1. Only extract what is EXPLICITLY stated. Do not infer or assume.
2. For experience_years, convert phrases like "5 years" → 5, "a decade" → 10.
3. For target_role, normalize to standard titles: "Senior ML Engineer", "Staff Data Scientist", etc.
4. For companies, use official names: "Google" not "goog", "Amazon" not "amzn".
5. If nothing extractable, return: {"no_extraction": true}

Respond with ONLY the JSON. No explanation."""


PROFILE_CONVERSATION_PROMPT = """You are AgentCoach's profile manager. Your job is to help the user
set up their profile for personalized interview preparation.

Current profile state:
{profile_summary}

Missing required fields: {missing_fields}

Instructions:
- If the user just shared profile info, acknowledge it warmly and confirm what you extracted.
- If required fields are missing, ask for them naturally (one at a time, not a list).
- If profile is complete, summarize and suggest next steps (study plan or mock interview).
- Be conversational, not form-like. Never ask more than one question per response.
- Keep responses to 2-4 sentences."""


class ProfileManagerAgent(BaseAgent):
    """Extracts and manages user profile from conversation.

    Two-phase operation:
    Phase 1 — EXTRACT: Parse user message for profile fields
    Phase 2 — RESPOND: Acknowledge extraction + ask for missing fields

    Interview key phrases:
    - "LLM-based information extraction from natural conversation"
    - "Two-phase: extract structured data, then generate conversational response"
    - "Hybrid approach: extract what we can, ask for what we can't"
    - "Profile validation with Pydantic — extraction errors caught at the boundary"
    """

    def __init__(self, llm: LLMGateway):
        super().__init__(
            name="profile_manager",
            description="Extracts and manages user profile from conversation",
            llm_tier="complex",  # Extraction needs GPT-4o quality
        )
        self.llm = llm

    def validate_input(self, state: AgentState) -> bool:
        """Profile agent works with any input — users can share info anytime."""
        return bool(state.get("current_input", "").strip())

    def build_prompt(self, state: AgentState) -> list[dict]:
        """Build extraction prompt. Called internally — not the main path."""
        messages = state.get("messages", [])[-5:]
        current_input = state.get("current_input", "")

        conversation_text = ""
        for msg in messages:
            conversation_text += f"{msg.get('role', 'unknown')}: {msg.get('content', '')}\n"
        conversation_text += f"user: {current_input}"

        return [
            {"role": "system", "content": PROFILE_EXTRACTION_PROMPT},
            {"role": "user", "content": f"Conversation:\n{conversation_text}\n\nExtract profile information."},
        ]

    def invoke(self, state: AgentState) -> AgentState:
        """Main execution: extract profile → update state → respond.

        ═══════════════════════════════════════════════════════════
        WHY TWO LLM CALLS (not one):
        ═══════════════════════════════════════════════════════════

        Call 1 (EXTRACTION): Uses a strict JSON-output prompt.
        - Temperature 0 for deterministic extraction
        - Structured output parsing with validation
        - If this fails, we still generate a response (graceful degradation)

        Call 2 (RESPONSE): Uses a conversational prompt.
        - Temperature 0.7 for natural-sounding response
        - Includes current profile state for context
        - Asks for missing fields if needed

        WHY NOT ONE CALL: Extraction quality degrades when you also ask
        the LLM to generate a conversational response. Separating concerns
        gives better extraction accuracy AND better conversation quality.

        Interview phrase: "I separate extraction from response generation.
        One focused LLM call for structured extraction at temperature 0,
        another for conversational response at temperature 0.7. Mixing
        both in one prompt degrades both — I learned this empirically."
        """
        start_time = datetime.now()

        # Load or create profile
        profile = self._load_profile(state)

        try:
            # ── PHASE 1: EXTRACT profile fields from user message ──
            extracted = self._extract_profile_fields(state)

            # ── PHASE 2: UPDATE profile with extracted fields ──
            changes_made = self._apply_extraction(profile, extracted)

            # ── PHASE 3: GENERATE conversational response ──
            response = self._generate_response(state, profile, changes_made)

            self._log_execution(state, start_time)

            return {
                **state,
                "agent_response": response,
                "user_profile": profile.model_dump(),
                "current_agent": "profile_manager",
                "is_complete": True,
            }

        except Exception as e:
            self._log_execution(state, start_time)
            return self.handle_error(state, e)

    def _load_profile(self, state: AgentState) -> UserProfile:
        """Load profile from state or create new one.

        WHY: State carries serialized profile dict. We deserialize
        into a Pydantic model for type-safe operations, then serialize
        back when we update state. This is the TypedDict ↔ Pydantic bridge.
        """
        profile_dict = state.get("user_profile")
        if profile_dict:
            try:
                return UserProfile.model_validate(profile_dict)
            except Exception:
                return UserProfile()  # Corrupted profile → start fresh
        return UserProfile()

    def _extract_profile_fields(self, state: AgentState) -> dict:
        """Phase 1: LLM extraction of profile fields.

        Uses GPT-4o at temperature 0 for deterministic extraction.
        Returns dict of extracted fields (only fields found in conversation).

        Interview phrase: "Temperature 0 for extraction — I want
        deterministic, reproducible results. The same input should
        always produce the same extraction. Temperature 0.7 is only
        for response generation where variety is desirable."
        """
        messages = self.build_prompt(state)

        raw_response = self.llm.chat(
            messages=messages,
            tier=ModelTier.COMPLEX,  # Extraction needs GPT-4o quality
            temperature=0,            # Deterministic for extraction
            max_tokens=300,
            agent_name=f"{self.name}_extraction",
        )

        return self._parse_extraction(raw_response)

    def _parse_extraction(self, raw_response: str) -> dict:
        """Parse extraction response with robust error handling.

        Interview phrase: "Extraction parsing with graceful degradation.
        If the LLM returns malformed JSON, we return an empty dict rather
        than crashing. The conversation still works — we just miss that
        extraction opportunity and ask the user directly next turn."
        """
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

        try:
            result = json.loads(cleaned)
            if result.get("no_extraction"):
                return {}
            return {k: v for k, v in result.items() if v is not None}
        except json.JSONDecodeError:
            return {}

    def _apply_extraction(self, profile: UserProfile, extracted: dict) -> list[str]:
        """Phase 2: Apply extracted fields to profile.

        Returns list of field names that were updated (for response generation).

        WHY SELECTIVE UPDATE (not blind overwrite):
        If user previously said "I'm targeting Google" and now says
        "I also want to try Amazon", we APPEND to target_companies,
        not overwrite. List fields use merge semantics.

        Interview phrase: "Selective profile update with merge semantics
        for list fields. New companies are appended, not overwritten.
        Scalar fields (name, experience) are replaced. This prevents
        information loss when users incrementally share their profile."
        """
        changes = []

        # Scalar fields — overwrite
        scalar_fields = ["name", "target_role", "experience_years", "preferred_study_hours_per_day"]
        for field in scalar_fields:
            if field in extracted and extracted[field]:
                current = getattr(profile, field)
                new_value = extracted[field]
                if current != new_value:
                    setattr(profile, field, new_value)
                    changes.append(field)

        # List fields — merge (append new, deduplicate)
        list_fields = ["target_companies", "tech_stack", "weak_areas_self_reported"]
        for field in list_fields:
            if field in extracted and extracted[field]:
                current = getattr(profile, field)
                new_items = extracted[field]
                if isinstance(new_items, list):
                    merged = list(set(current + new_items))
                    if merged != current:
                        setattr(profile, field, merged)
                        changes.append(field)

        # Dict fields — merge
        if "interview_dates" in extracted and extracted["interview_dates"]:
            for company, date in extracted["interview_dates"].items():
                if company not in profile.interview_dates:
                    profile.interview_dates[company] = date
                    changes.append("interview_dates")

        if changes:
            profile.updated_at = datetime.now()
            profile.profile_version += 1

        return changes

    def _generate_response(
        self, state: AgentState, profile: UserProfile, changes: list[str]
    ) -> str:
        """Phase 3: Generate conversational response.

        WHY CONTEXT-AWARE RESPONSE:
        The response depends on:
        1. What was just extracted (acknowledge it)
        2. What's still missing (ask for it)
        3. Whether profile is complete (suggest next steps)

        Interview phrase: "The response is context-aware — it acknowledges
        what was extracted, asks for the most important missing field, or
        suggests next steps if the profile is complete. Never asks for
        more than one field at a time to avoid overwhelming the user."
        """
        # Build profile summary
        profile_summary = self._build_profile_summary(profile)

        # Determine missing required fields
        missing = []
        if not profile.name:
            missing.append("name")
        if not profile.target_role:
            missing.append("target role")
        if profile.experience_years == 0:
            missing.append("years of experience")

        prompt = PROFILE_CONVERSATION_PROMPT.format(
            profile_summary=profile_summary,
            missing_fields=", ".join(missing) if missing else "none — profile is complete!",
        )

        # Include what was just extracted for acknowledgment
        context = ""
        if changes:
            context = f"\nJust extracted from user's message: {', '.join(changes)}"

        messages_for_llm = [
            {"role": "system", "content": prompt + context},
        ]

        # Add recent conversation
        recent = state.get("messages", [])[-3:]
        messages_for_llm.extend(recent)
        messages_for_llm.append({
            "role": "user",
            "content": state.get("current_input", ""),
        })

        response = self.llm.chat(
            messages=messages_for_llm,
            tier=ModelTier.SIMPLE,  # Response generation is simpler than extraction
            temperature=0.7,
            max_tokens=300,
            agent_name=f"{self.name}_response",
        )

        return response

    def _build_profile_summary(self, profile: UserProfile) -> str:
        """Build human-readable profile summary for the response prompt."""
        parts = []
        if profile.name:
            parts.append(f"Name: {profile.name}")
        if profile.target_role:
            parts.append(f"Target role: {profile.target_role}")
        if profile.experience_years > 0:
            parts.append(f"Experience: {profile.experience_years} years")
        if profile.target_companies:
            parts.append(f"Target companies: {', '.join(profile.target_companies)}")
        if profile.tech_stack:
            parts.append(f"Tech stack: {', '.join(profile.tech_stack)}")
        if profile.interview_dates:
            dates = [f"{c}: {d}" for c, d in profile.interview_dates.items()]
            parts.append(f"Interview dates: {', '.join(dates)}")

        return "\n".join(parts) if parts else "Empty — no information collected yet."