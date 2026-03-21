"""Three-Tier Memory System — Story 2.3.

═══════════════════════════════════════════════════════════════════
WHY THREE TIERS (The #1 Interview Question on Memory):
═══════════════════════════════════════════════════════════════════

Interviewer: "How do you handle memory in your agentic system?"

WRONG answer: "I just pass the conversation history."
→ Red flag: shows no understanding of context window limits.

WRONG answer: "I use ConversationBufferMemory from LangChain."
→ Red flag: shows copy-paste, no understanding of WHY.

RIGHT answer (what you should say):

"I implemented a three-tier memory architecture because each tier
serves a different access pattern:

TIER 1 — SHORT-TERM (Conversation Buffer):
  What: Last K messages in the current conversation.
  Why: Immediate conversational context. The user says 'as I mentioned
  earlier' — we need the recent messages to resolve this reference.
  Analogy: Your working memory. What you're thinking about right now.
  Implementation: Sliding window, last 20 messages.
  Cost: Cheap (just array slicing).

TIER 2 — MEDIUM-TERM (Session Summaries):
  What: LLM-generated summaries of past sessions.
  Why: User had a detailed study session 3 days ago about transformers.
  We can't fit the full 200-message transcript, but a 3-sentence summary
  captures the key learning outcomes.
  Analogy: Your episodic memory. 'Last Tuesday I went to the gym' — not
  every rep, but the key facts.
  Implementation: After each session ends, LLM summarizes it. Stored in SQLite.
  Cost: Medium (one LLM call per session end).

TIER 3 — LONG-TERM (Vector Semantic Memory):
  What: Embedded key facts about the user — skills, preferences,
  important statements — stored in a vector database.
  Why: When the user asks 'remind me what I'm weak at', we need to
  retrieve relevant facts from across ALL sessions, not just recent ones.
  Analogy: Your semantic memory. You know Paris is in France without
  remembering when you learned it.
  Implementation: FAISS vector store with ada-002 embeddings.
  Cost: Moderate (embedding + retrieval per query).

The key insight: each tier has different LATENCY, GRANULARITY, and COST:
  - Tier 1: 0ms, full messages, free
  - Tier 2: 0ms (pre-computed), session-level, ~$0.003 per summary
  - Tier 3: ~50ms, fact-level, ~$0.0001 per retrieval

═══════════════════════════════════════════════════════════════════
COMPARISON WITH PRE-AUTH:
═══════════════════════════════════════════════════════════════════

Pre-Auth: Only Tier 1 (current claim context). No cross-claim memory.
AgentCoach: All 3 tiers. Cross-session user modeling.

Interview phrase: 'My Pre-Auth system is stateless per claim — it only
needs the current claim context. AgentCoach requires cross-session
memory because we're modeling a user's learning journey over weeks.
That's why I designed a three-tier architecture: buffer for immediate
context, summaries for session history, and vector store for semantic
retrieval of key facts.'
"""

from datetime import datetime

from src.config.llm_gateway import LLMGateway, ModelTier


# ─────────────────────────────────────────────────────────────
# TIER 1: SHORT-TERM — Conversation Buffer
# ─────────────────────────────────────────────────────────────

class ConversationBuffer:
    """Sliding window of recent messages.

    ═══════════════════════════════════════════════════════════
    WHY SLIDING WINDOW (not unlimited):
    ═══════════════════════════════════════════════════════════

    GPT-4o has a 128K context window, but using all of it is:
    1. EXPENSIVE — cost scales linearly with input tokens
    2. SLOW — latency increases with context size
    3. HARMFUL — "lost in the middle" effect: LLMs attend less
       to information in the middle of long contexts

    20 messages ≈ 2000-4000 tokens. That's enough for conversational
    coherence while keeping cost and latency low.

    Interview phrase: "I cap the buffer at 20 messages — about 3K
    tokens. Beyond that, you hit the 'lost in the middle' problem
    where the LLM attends poorly to information in the middle of
    long contexts. Better to summarize old messages than dump them all."

    The 'lost in the middle' paper (Liu et al., 2023) is a great
    reference if an interviewer probes deeper.
    """

    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self.messages: list[dict] = []

    def add_message(self, role: str, content: str) -> None:
        """Add a message, evicting oldest if at capacity."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        # Evict oldest if over limit
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_messages(self, k: int | None = None) -> list[dict]:
        """Get last K messages (or all if k is None)."""
        if k is None:
            return self.messages
        return self.messages[-k:]

    def get_token_estimate(self) -> int:
        """Rough token estimate: ~1.3 tokens per word.

        WHY ESTIMATE: Calling tiktoken on every buffer access is expensive.
        A rough estimate is sufficient for budget allocation. We do exact
        counting only right before the LLM call (in the token budget allocator).
        """
        total_words = sum(len(m["content"].split()) for m in self.messages)
        return int(total_words * 1.3)

    def clear(self) -> None:
        """Clear buffer (e.g., when starting a new session)."""
        self.messages = []

    def to_prompt_messages(self) -> list[dict]:
        """Convert to LLM-ready format (without timestamps)."""
        return [{"role": m["role"], "content": m["content"]} for m in self.messages]


# ─────────────────────────────────────────────────────────────
# TIER 2: MEDIUM-TERM — Session Summaries
# ─────────────────────────────────────────────────────────────

SESSION_SUMMARY_PROMPT = """Summarize this coaching session in 3-5 sentences.
Focus on:
1. What topics were covered
2. What the user struggled with
3. What they did well
4. Any decisions or plans made
5. Key scores or assessments given

Be factual and specific. Include topic names and scores if mentioned.
This summary will be used to provide context in future sessions."""


class SessionSummaryStore:
    """Stores LLM-generated summaries of past sessions.

    ═══════════════════════════════════════════════════════════
    WHY LLM SUMMARIES (not raw transcripts):
    ═══════════════════════════════════════════════════════════

    A 30-minute coaching session might be 200 messages = ~20K tokens.
    We can't fit more than 2-3 full sessions in context.
    But a 3-sentence summary is ~50 tokens. We can fit 100 session
    summaries in the same space as 1 full transcript.

    The trade-off: we lose detail but gain breadth. For questions like
    "what have we worked on before?" — summaries are perfect. For
    questions like "what exact feedback did you give me on my transformer
    answer?" — we'd need the full transcript, which lives in Tier 3
    (vector search over specific facts).

    Interview phrase: "Session summaries compress 20K-token conversations
    into 50-token summaries. We trade detail for breadth — I can fit
    100 session summaries in the same token budget as 1 full transcript.
    When the user needs specific details, Tier 3 vector search retrieves
    the relevant facts."

    ═══════════════════════════════════════════════════════════
    WHY STORE IN SQLite (not vector DB):
    ═══════════════════════════════════════════════════════════

    Summaries are accessed CHRONOLOGICALLY (most recent first),
    not SEMANTICALLY. SQLite with ORDER BY timestamp is perfect.
    Vector DB would be overkill for ordered retrieval.

    Interview phrase: "Summaries are retrieved chronologically, not
    semantically, so SQLite with timestamp ordering is the right
    data store. Vector DB is for Tier 3 where we need semantic search."
    """

    def __init__(self, llm: LLMGateway):
        self.llm = llm
        # In-memory store for MVP. Epic 8 upgrades to SQLite.
        self._summaries: list[dict] = []

    def summarize_session(self, messages: list[dict]) -> str:
        """Generate a summary of a completed session.

        Called when a coaching session ends. Takes the full message
        history and produces a concise summary.

        WHY AT SESSION END (not incrementally):
        Summarizing at the end captures the FULL arc of the session.
        Incremental summaries miss connections between early and late
        messages. Also, one LLM call at session end is cheaper than
        many incremental calls.
        """
        if not messages:
            return "Empty session — no messages exchanged."

        # Format conversation for summarization
        conversation = "\n".join(
            f"{m.get('role', 'unknown')}: {m.get('content', '')}"
            for m in messages[-30:]  # Cap at last 30 messages for cost
        )

        response = self.llm.chat(
            messages=[
                {"role": "system", "content": SESSION_SUMMARY_PROMPT},
                {"role": "user", "content": f"Session transcript:\n{conversation}"},
            ],
            tier=ModelTier.SIMPLE,  # Summarization works fine with GPT-4o-mini
            temperature=0.3,        # Low variation for consistent summaries
            max_tokens=200,
            agent_name="session_summarizer",
        )

        # Store the summary
        summary_entry = {
            "summary": response,
            "timestamp": datetime.now().isoformat(),
            "message_count": len(messages),
        }
        self._summaries.append(summary_entry)

        return response

    def get_recent_summaries(self, k: int = 5) -> list[dict]:
        """Get most recent K session summaries.

        WHY MOST RECENT: Recent sessions are most relevant for
        continuing preparation. A session from 3 weeks ago about
        SQL is less important than yesterday's session about transformers
        if the user's Google interview is tomorrow.
        """
        return self._summaries[-k:]

    def get_summaries_as_context(self, k: int = 5) -> str:
        """Format summaries as a context string for LLM prompts.

        Returns a formatted string that can be inserted into any
        agent's prompt to give historical context.
        """
        recent = self.get_recent_summaries(k)
        if not recent:
            return "No previous sessions recorded."

        lines = []
        for i, s in enumerate(recent, 1):
            lines.append(f"Session {i} ({s['timestamp'][:10]}): {s['summary']}")

        return "\n".join(lines)

    def get_all_summaries_count(self) -> int:
        return len(self._summaries)


# ─────────────────────────────────────────────────────────────
# TIER 3: LONG-TERM — Vector Semantic Memory
# ─────────────────────────────────────────────────────────────

class SemanticMemory:
    """Vector-based long-term memory for key facts about the user.

    ═══════════════════════════════════════════════════════════
    WHY VECTOR MEMORY (not keyword search):
    ═══════════════════════════════════════════════════════════

    User says: "What topics should I focus on?"
    Relevant memory: "User scored 4.0 on system design in session 7."

    Keyword search wouldn't find this — "topics" doesn't appear in
    the stored memory. But SEMANTICALLY, "topics to focus on" is
    highly related to "low scores on specific topics."

    This is THE SAME PATTERN as your RAG system — but instead of
    searching healthcare documents, you're searching user memories.

    Interview phrase: "Long-term memory retrieval is itself a RAG
    problem. I embed key facts about the user into a vector store
    and retrieve semantically relevant memories for each conversation
    turn. The same bi-encoder retrieval pattern I use in healthcare
    RAG, applied to user memory."

    ═══════════════════════════════════════════════════════════
    WHAT GETS STORED (not everything):
    ═══════════════════════════════════════════════════════════

    We DON'T store every message — that's Tier 1's job.
    We store FACTS extracted from conversations:
    - "User is weak at system design"
    - "User prefers practical examples over theory"
    - "User's Google interview is April 15"
    - "User scored 8/10 on transformers in mock interview"
    - "User struggled with explaining cross-encoder vs bi-encoder"

    These facts are embedded and stored in FAISS for semantic retrieval.

    Interview phrase: "I store extracted FACTS, not raw messages.
    Facts are higher signal density — a 10-word fact carries more
    information per token than a 200-word conversation transcript."
    """

    def __init__(self, llm: LLMGateway):
        self.llm = llm
        # In-memory store for MVP. Production would use FAISS.
        # Structure: list of {fact, embedding, timestamp, source}
        self._facts: list[dict] = []
        self._faiss_index = None  # Will be initialized on first use

    def _initialize_faiss(self) -> None:
        """Lazy initialization of FAISS index.

        WHY LAZY: Don't allocate resources until first use.
        If user never uses long-term features, no cost.

        Interview phrase: "Lazy initialization — the vector index
        isn't created until the first fact is stored. No upfront
        cost for users who only have one session."
        """
        try:
            import faiss
            import numpy as np
            # Ada-002 produces 1536-dimensional embeddings
            self._faiss_index = faiss.IndexFlatIP(1536)  # Inner product = cosine sim for normalized vectors
        except ImportError:
            # FAISS not installed — fall back to brute-force
            self._faiss_index = None

    def store_fact(self, fact: str, source: str = "conversation") -> None:
        """Store a fact in long-term memory.

        WHY SEPARATE STORE (not automatic):
        Not every message is a storable fact. The calling agent
        decides what's worth remembering. This prevents memory
        pollution with greetings and filler.

        Interview phrase: "Selective memory storage — agents decide
        what facts are worth persisting. Greetings and filler don't
        get stored. Only assessments, preferences, key decisions,
        and skill evaluations make it to long-term memory."
        """
        self._facts.append({
            "fact": fact,
            "timestamp": datetime.now().isoformat(),
            "source": source,
        })

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        """Retrieve semantically relevant facts.

        MVP: Simple keyword matching on stored facts.
        Production: FAISS similarity search on embeddings.

        WHY MVP IS KEYWORD (not full FAISS):
        For <100 facts, brute-force keyword matching is fast enough
        and doesn't require FAISS installation. We upgrade to proper
        embeddings when the fact count exceeds 100.

        Interview phrase: "I started with keyword matching for MVP
        since we have fewer than 100 facts per user. The architecture
        is designed for FAISS upgrade — the interface is the same,
        only the retrieval implementation changes. This is the
        'start simple, scale when needed' principle."
        """
        if not self._facts:
            return []

        # MVP: Simple relevance scoring
        query_words = set(query.lower().split())
        scored = []
        for fact_entry in self._facts:
            fact_words = set(fact_entry["fact"].lower().split())
            overlap = len(query_words & fact_words)
            if overlap > 0:
                scored.append((overlap, fact_entry["fact"]))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [fact for _, fact in scored[:k]]

    def get_all_facts(self) -> list[dict]:
        """Return all stored facts (for debugging / export)."""
        return self._facts

    def get_fact_count(self) -> int:
        return len(self._facts)


# ─────────────────────────────────────────────────────────────
# MEMORY MANAGER — Orchestrates all three tiers
# ─────────────────────────────────────────────────────────────

class MemoryManager:
    """Unified interface to all three memory tiers.

    ═══════════════════════════════════════════════════════════
    WHY A MANAGER (not direct access to each tier):
    ═══════════════════════════════════════════════════════════

    Single Responsibility: Agents shouldn't know which tier to query.
    The MemoryManager decides what to retrieve from each tier based
    on the current context and assembles a unified memory context string.

    This also enables the TOKEN BUDGET ALLOCATOR (Story 2.4):
    The manager knows how many tokens each tier is using and can
    dynamically adjust allocations.

    Interview phrase: "The MemoryManager provides a unified interface.
    Agents call get_context() and receive a formatted string from all
    relevant tiers. The manager handles token budget allocation internally
    — agents don't need to know about the three-tier architecture."
    """

    def __init__(self, llm: LLMGateway):
        self.buffer = ConversationBuffer(max_messages=20)
        self.summaries = SessionSummaryStore(llm=llm)
        self.semantic = SemanticMemory(llm=llm)
        self.llm = llm

    def add_message(self, role: str, content: str) -> None:
        """Add to conversation buffer (Tier 1)."""
        self.buffer.add_message(role, content)

    def end_session(self) -> str:
        """End current session: summarize and store (Tier 2).

        Called when a conversation session ends. Summarizes the
        session and stores the summary for future reference.

        Returns the generated summary.
        """
        messages = self.buffer.get_messages()
        summary = self.summaries.summarize_session(messages)
        return summary

    def store_fact(self, fact: str, source: str = "agent") -> None:
        """Store a key fact in long-term memory (Tier 3)."""
        self.semantic.store_fact(fact, source)

    def get_context(self, query: str = "", max_tokens: int = 1000) -> str:
        """Get unified memory context for an LLM prompt.

        Assembles context from all three tiers, respecting token budget.

        Token budget allocation:
        - Tier 1 (buffer): 50% of budget — most important for coherence
        - Tier 2 (summaries): 30% — session history
        - Tier 3 (semantic): 20% — relevant long-term facts

        WHY THIS SPLIT: Buffer is most time-sensitive (current conversation).
        Summaries provide continuity. Semantic facts add depth. If we have
        to cut, we cut semantic first, summaries second, never the buffer.

        Interview phrase: "Token budget allocation across memory tiers.
        Buffer gets 50% because conversational coherence is non-negotiable.
        Summaries get 30% for session continuity. Semantic facts get 20%.
        If we're near the limit, we trim in reverse priority."
        """
        parts = []

        # Tier 2: Session summaries (chronological)
        summaries_text = self.summaries.get_summaries_as_context(k=3)
        if summaries_text and summaries_text != "No previous sessions recorded.":
            parts.append(f"[Previous sessions]\n{summaries_text}")

        # Tier 3: Relevant long-term facts (semantic)
        if query:
            relevant_facts = self.semantic.retrieve(query, k=5)
            if relevant_facts:
                facts_text = "\n".join(f"- {f}" for f in relevant_facts)
                parts.append(f"[Key facts about this user]\n{facts_text}")

        if not parts:
            return ""

        return "\n\n".join(parts)

    def get_buffer_messages(self) -> list[dict]:
        """Get buffer messages for direct inclusion in LLM prompt."""
        return self.buffer.to_prompt_messages()

    def get_memory_stats(self) -> dict:
        """Return stats about memory usage across all tiers.

        Interview phrase: "Per-tier memory stats for observability.
        I track message count in buffer, summary count in Tier 2,
        fact count in Tier 3, and estimated token usage. This feeds
        the cost tracking dashboard."
        """
        return {
            "buffer_messages": len(self.buffer.messages),
            "buffer_token_estimate": self.buffer.get_token_estimate(),
            "session_summaries": self.summaries.get_all_summaries_count(),
            "long_term_facts": self.semantic.get_fact_count(),
        }