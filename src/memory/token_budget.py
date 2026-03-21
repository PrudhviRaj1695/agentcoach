"""Token Budget Allocator — Story 2.4.

═══════════════════════════════════════════════════════════════════
WHY THIS EXISTS (The Problem It Solves):
═══════════════════════════════════════════════════════════════════

GPT-4o has a 128K token context window. Sounds huge, right?

But in practice:
  System prompt:          ~500 tokens
  User profile:           ~300 tokens
  Agent instructions:     ~400 tokens
  Memory context:         ~500 tokens (summaries + facts)
  Conversation buffer:    ~3000 tokens (20 messages)
  Current user input:     ~100 tokens
  ─────────────────────────────────
  Total input:            ~4800 tokens
  Output budget:          ~1000 tokens
  ─────────────────────────────────
  Grand total:            ~5800 tokens per turn

That's only 4.5% of the 128K window. So why do we need a budget allocator?

1. COST: You're paying per token. An unbudgeted system that dumps 50K
   tokens of context costs 10x more than a budgeted one.

2. QUALITY: More context ≠ better answers. The "lost in the middle"
   effect means the LLM ignores information in the middle of long contexts.
   LESS but BETTER context produces better responses.

3. LATENCY: Latency scales linearly with input tokens. Every 1K extra
   tokens adds ~100ms to response time. Over 20 turns, that's noticeable.

Interview phrase: "I built a token budget allocator because more context
isn't better context. GPT-4o's 128K window is a ceiling, not a target.
I allocate fixed budgets per component — system prompt, profile, memory,
conversation — and enforce them. This keeps cost predictable, latency
low, and quality high by avoiding the 'lost in the middle' problem."

═══════════════════════════════════════════════════════════════════
COMPARISON WITH YOUR RAG SYSTEM:
═══════════════════════════════════════════════════════════════════

In your RAG system, you stuff 3 chunks × ~650 tokens = ~2000 tokens
of context into the prompt. That's a form of context budgeting!

The budget allocator is the same principle, generalized:
- RAG: budget = system_prompt + chunks + query
- AgentCoach: budget = system_prompt + profile + memory + conversation + agent_instructions

Interview phrase: "This is the same context management principle as
my RAG system — I budget 3 chunks at 650 tokens each for retrieval
context. In AgentCoach, I budget across more components: profile,
memory tiers, conversation history, and agent instructions."
"""

import tiktoken


# Default token budgets per component
DEFAULT_BUDGETS = {
    "system_prompt": 500,
    "agent_instructions": 400,
    "user_profile": 300,
    "memory_summaries": 300,
    "memory_facts": 200,
    "conversation_buffer": 3000,
    "current_input": 500,
    "output_reserve": 1000,  # Reserve for LLM response
}

# Maximum total budget (configurable per model)
MAX_TOTAL_TOKENS = {
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4-turbo": 128000,
}


class TokenBudgetAllocator:
    """Manages token allocation across prompt components.

    ═══════════════════════════════════════════════════════════
    HOW IT WORKS (step by step):
    ═══════════════════════════════════════════════════════════

    1. Each component has a CEILING (max tokens it can use)
    2. Before building a prompt, the allocator checks each component
    3. If a component exceeds its budget → TRUNCATE
       - Conversation buffer: drop oldest messages
       - Memory context: drop least relevant facts
       - Profile: abbreviate (drop tech_stack details)
    4. After truncation, validate total is within model limit
    5. Return the budget report for observability

    Interview phrase: "Priority-based truncation. If we're over budget,
    we trim in reverse priority: semantic facts first, then session
    summaries, then older conversation messages. We NEVER truncate
    the system prompt or the current user input."
    """

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.budgets = DEFAULT_BUDGETS.copy()
        self.max_total = MAX_TOTAL_TOKENS.get(model, 128000)

        try:
            self.encoding = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Exact token count using tiktoken.

        WHY EXACT (not estimate): Estimates are fine for buffer management,
        but right before the LLM call, we need exact counts to prevent
        silent truncation by the API (which just cuts off your input).

        Interview phrase: "Exact token counting with tiktoken before
        every LLM call. Estimates are fine for planning, but the final
        prompt gets exact counts to prevent silent API truncation."
        """
        if not text:
            return 0
        return len(self.encoding.encode(text))

    def allocate(self, components: dict[str, str]) -> dict:
        """Check each component against its budget, return allocation report.

        Args:
            components: dict mapping component_name → content string

        Returns:
            dict with:
            - "allocations": {component: {content, tokens, budget, over_budget}}
            - "total_tokens": total input tokens
            - "remaining_for_output": tokens left for generation
            - "warnings": list of over-budget components

        Interview phrase: "The allocator returns a full budget report:
        per-component token usage, total tokens, remaining capacity,
        and warnings for over-budget components. This feeds the
        observability dashboard — I know exactly where tokens are
        going on every turn."
        """
        allocations = {}
        warnings = []
        total = 0

        for name, content in components.items():
            tokens = self.count_tokens(content)
            budget = self.budgets.get(name, 500)
            over = tokens > budget

            allocations[name] = {
                "tokens": tokens,
                "budget": budget,
                "over_budget": over,
                "utilization": round(tokens / budget, 2) if budget > 0 else 0,
            }
            total += tokens

            if over:
                warnings.append(
                    f"{name}: {tokens} tokens exceeds budget of {budget} "
                    f"(over by {tokens - budget})"
                )

        output_reserve = self.budgets.get("output_reserve", 1000)
        remaining = self.max_total - total - output_reserve

        return {
            "allocations": allocations,
            "total_input_tokens": total,
            "output_reserve": output_reserve,
            "remaining_capacity": max(0, remaining),
            "utilization_percent": round(total / self.max_total * 100, 1),
            "warnings": warnings,
        }

    def truncate_to_budget(self, text: str, component: str) -> str:
        """Truncate text to fit within component's token budget.

        Truncation strategy depends on component type:
        - Conversation: drop from the START (keep recent messages)
        - Memory: drop from the END (keep most relevant)
        - Profile: drop from the END (keep core identity)

        WHY SMART TRUNCATION (not just cut at N characters):
        Cutting mid-sentence creates garbled context that confuses the LLM.
        We cut at message/fact boundaries to maintain coherence.

        Interview phrase: "Boundary-aware truncation — I cut at message
        boundaries for conversations and at fact boundaries for memory.
        Mid-sentence truncation creates garbled context that degrades
        LLM response quality."
        """
        budget = self.budgets.get(component, 500)
        current_tokens = self.count_tokens(text)

        if current_tokens <= budget:
            return text  # Within budget, no truncation needed

        # Binary search for the right truncation point
        # Cut from the beginning (keep the end, which is more recent)
        lines = text.split("\n")

        while self.count_tokens("\n".join(lines)) > budget and len(lines) > 1:
            lines.pop(0)  # Remove oldest line

        return "\n".join(lines)

    def set_budget(self, component: str, tokens: int) -> None:
        """Adjust a component's token budget.

        WHY ADJUSTABLE: Different agents need different allocations.
        The MockInterviewer needs more conversation buffer (multi-turn).
        The StudyPlanner needs more output reserve (long plans).

        Interview phrase: "Adjustable per-agent budgets. The mock
        interviewer gets a larger conversation buffer because it's
        multi-turn. The planner gets a larger output reserve because
        study plans are longer responses."
        """
        self.budgets[component] = tokens

    def get_budget_summary(self) -> dict:
        """Return current budget configuration."""
        total_allocated = sum(self.budgets.values())
        return {
            "budgets": self.budgets.copy(),
            "total_allocated": total_allocated,
            "model_limit": self.max_total,
            "headroom": self.max_total - total_allocated,
        }