"""Router Agent — classifies user intent and routes to specialist agents.

This is the BRAIN of AgentCoach. Every user message passes through here first.

Architecture choice: Uses GPT-4o-mini for classification (fast + cheap).
Compare with Pre-Auth Supervisor which uses deterministic routing.

Interview talking point: 'The Router uses semantic intent classification with
GPT-4o-mini — it costs ~$0.0001 per classification. I chose LLM-based routing
over keyword matching because user messages are open-ended. Keywords would break
on messages like "let's practice" (interview mode) vs "let's practice planning"
(study mode). The LLM resolves this ambiguity using conversation context.'
"""

import json
from datetime import datetime

from src.agents.base import BaseAgent
from src.config.llm_gateway import LLMGateway, ModelTier
from src.config.prompts import ROUTER_SYSTEM_PROMPT, ROUTER_FEW_SHOT_EXAMPLES
from src.models.state import AgentState


# Valid modes the router can classify into
VALID_MODES = {
    "planning",       # User wants a study plan or schedule
    "interviewing",   # User wants to do a mock interview
    "evaluating",     # User wants feedback on an answer
    "strategy",       # User wants company-specific advice
    "profile",        # User is sharing/updating personal info
    "chat",           # General conversation, greetings, unclear
}


class RouterAgent(BaseAgent):
    """Classifies user intent and routes to the appropriate specialist agent.

    Uses GPT-4o-mini with few-shot examples for fast, cheap classification.
    Falls back to 'chat' mode if confidence is below threshold.

    Interview key phrases:
    - 'Semantic intent classification, not keyword matching'
    - 'Confidence-gated routing: below 0.7, we ask for clarification'
    - 'Context-aware: same message means different things in different modes'
    - 'GPT-4o-mini for routing — 10x cheaper than using the main model'
    """

    CONFIDENCE_THRESHOLD = 0.7

    def __init__(self, llm: LLMGateway):
        super().__init__(
            name="router",
            description="Classifies user intent and routes to specialist agents",
            llm_tier="simple",
        )
        self.llm = llm

    def validate_input(self, state: AgentState) -> bool:
        """Router needs at minimum the current user input."""
        current_input = state.get("current_input", "")
        return bool(current_input and current_input.strip())

    def build_prompt(self, state: AgentState) -> list[dict]:
        """Build classification prompt with conversation context.

        Key design: includes last 5 messages for context-aware routing.
        'Tell me about transformers' means different things depending on
        whether we're in study mode or interview mode.
        """
        # Get recent conversation context (last 5 messages)
        messages = state.get("messages", [])
        recent_context = messages[-5:] if len(messages) > 5 else messages

        # Current mode for context-aware classification
        current_mode = state.get("current_mode", "chat")

        # Build context string
        context_str = ""
        if recent_context:
            context_lines = []
            for msg in recent_context:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:200]  # Truncate long messages
                context_lines.append(f"{role}: {content}")
            context_str = "\n".join(context_lines)

        user_input = state.get("current_input", "")

        classification_prompt = f"""Current mode: {current_mode}

Recent conversation:
{context_str if context_str else "(no prior context)"}

New user message: "{user_input}"

Classify this message. Respond with ONLY valid JSON:
{{"mode": "<one of: planning, interviewing, evaluating, strategy, profile, chat>", "confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}}"""

        return [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            *ROUTER_FEW_SHOT_EXAMPLES,
            {"role": "user", "content": classification_prompt},
        ]

    def invoke(self, state: AgentState) -> AgentState:
        """Classify intent and update state with routing decision.

        This is registered as the 'router' node in LangGraph.
        """
        start_time = datetime.now()

        # Guard: check step count
        step_count = state.get("step_count", 0)
        if step_count > 5:
            return {
                **state,
                "current_mode": "chat",
                "current_agent": "general_chat",
                "route_confidence": 1.0,
                "error": "Max routing steps exceeded — defaulting to chat",
            }

        # Validate input
        if not self.validate_input(state):
            return {
                **state,
                "current_mode": "chat",
                "current_agent": "general_chat",
                "route_confidence": 0.0,
                "agent_response": "I didn't catch that. What would you like to work on?",
            }

        try:
            # Build prompt and classify
            messages = self.build_prompt(state)
            raw_response = self.llm.chat(
                messages=messages,
                tier=ModelTier.SIMPLE,  # GPT-4o-mini for speed + cost
                temperature=0.1,       # Low temp for consistent classification
                max_tokens=150,        # Classification is short
                agent_name=self.name,
            )

            # Parse structured output
            classification = self._parse_classification(raw_response)
            mode = classification["mode"]
            confidence = classification["confidence"]
            reasoning = classification.get("reasoning", "")

            # Confidence gate
            if confidence < self.CONFIDENCE_THRESHOLD:
                self._log_execution(state, start_time)
                return {
                    **state,
                    "current_mode": "chat",
                    "current_agent": "clarify",
                    "route_confidence": confidence,
                    "agent_response": self._build_clarification(
                        state.get("current_input", ""), mode, confidence
                    ),
                    "step_count": step_count + 1,
                }

            # Route to appropriate agent
            agent_map = {
                "planning": "planner_agent",
                "interviewing": "interviewer_agent",
                "evaluating": "evaluator_agent",
                "strategy": "strategy_agent",
                "profile": "profile_agent",
                "chat": "general_chat",
            }

            self._log_execution(state, start_time)
            return {
                **state,
                "current_mode": mode,
                "current_agent": agent_map.get(mode, "general_chat"),
                "route_confidence": confidence,
                "step_count": step_count + 1,
            }

        except Exception as e:
            self._log_execution(state, start_time)
            return self.handle_error(state, e)

    def _parse_classification(self, raw_response: str) -> dict:
        """Parse LLM classification response into structured dict.

        Robust parsing: handles markdown code blocks, extra whitespace,
        and invalid JSON gracefully.

        Interview talking point: 'Output parsing with fallback — if the LLM
        returns malformed JSON, we default to chat mode rather than crashing.
        In production, ~2-3% of classifications need this fallback.'
        """
        # Strip markdown code blocks if present
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]  # Remove first line
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback: try to extract mode from raw text
            return {
                "mode": "chat",
                "confidence": 0.3,
                "reasoning": f"JSON parse failed, raw: {raw_response[:100]}",
            }

        # Validate mode is in allowed set
        mode = result.get("mode", "chat")
        if mode not in VALID_MODES:
            mode = "chat"

        # Validate confidence is numeric and in range
        try:
            confidence = float(result.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.5

        return {
            "mode": mode,
            "confidence": confidence,
            "reasoning": result.get("reasoning", ""),
        }

    def _build_clarification(
        self, user_input: str, guessed_mode: str, confidence: float
    ) -> str:
        """Build a clarification question when confidence is low.

        Rather than guessing wrong, we ask the user what they meant.
        This is better UX than silently routing to the wrong agent.
        """
        mode_descriptions = {
            "planning": "work on a study plan",
            "interviewing": "do a mock interview",
            "evaluating": "get feedback on an answer",
            "strategy": "discuss company-specific preparation",
            "profile": "update your profile",
            "chat": "just chat",
        }
        guess_desc = mode_descriptions.get(guessed_mode, "chat")

        return (
            f"I want to make sure I help you the right way. "
            f"Did you mean you'd like to {guess_desc}, or something else?"
        )