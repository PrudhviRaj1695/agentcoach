"""LangGraph orchestration — wires all agents into a stateful graph.

This is the HEART of the system. It defines:
1. The StateGraph with all nodes
2. Conditional edges (Router → specialist agents)
3. State transitions and terminal conditions
4. The compiled graph ready for execution

Interview talking point: 'I use LangGraph's StateGraph with conditional edges
for orchestration. The Router node classifies intent, then conditional edges
route to the correct specialist agent based on the classification. Each agent
reads from and writes to a shared typed state — no information loss during
handoffs. The graph includes cycle detection with a max step count to prevent
infinite routing loops.'

Key architectural difference from Pre-Auth:
- Pre-Auth: Linear pipeline (retrieve → analyze → validate → respond)
- AgentCoach: Dynamic routing (Router classifies → routes to any agent)
"""

from langgraph.graph import END, StateGraph

from src.agents.router import RouterAgent
from src.config.llm_gateway import LLMGateway, ModelTier
from src.config.prompts import GENERAL_CHAT_SYSTEM_PROMPT
from src.models.state import AgentState


def create_general_chat_node(llm: LLMGateway):
    """Create a simple chat node for general conversation.

    This handles: greetings, unclear intent, meta-questions about the system.
    Uses GPT-4o-mini since these are simple responses.
    """

    def general_chat(state: AgentState) -> AgentState:
        messages = [
            {"role": "system", "content": GENERAL_CHAT_SYSTEM_PROMPT},
        ]

        # Add recent conversation for context
        recent = state.get("messages", [])[-5:]
        messages.extend(recent)

        # Add current input
        current_input = state.get("current_input", "")
        if current_input:
            messages.append({"role": "user", "content": current_input})

        response = llm.chat(
            messages=messages,
            tier=ModelTier.SIMPLE,
            temperature=0.7,
            max_tokens=500,
            agent_name="general_chat",
        )

        return {
            **state,
            "agent_response": response,
            "current_agent": "general_chat",
            "is_complete": True,
        }

    return general_chat


def create_clarify_node(llm: LLMGateway):
    """Create a clarification node for low-confidence routing.

    When the Router isn't confident, this node asks the user to clarify
    rather than guessing wrong. Better UX than silent misrouting.
    """

    def clarify(state: AgentState) -> AgentState:
        # The router already set agent_response with clarification question
        return {
            **state,
            "is_complete": True,
            "current_agent": "clarify",
        }

    return clarify


def create_placeholder_node(agent_name: str):
    """Placeholder for agents we haven't built yet (Epics 2-5).

    Returns a helpful message telling the user the feature is coming.
    This lets us test the full routing flow before all agents exist.

    Interview talking point: 'I scaffolded placeholder nodes for every agent
    from day one. This let me test the full routing and handoff flow before
    implementing each specialist. Incremental integration, not big-bang.'
    """

    def placeholder(state: AgentState) -> AgentState:
        agent_descriptions = {
            "profile_agent": "Profile management (capturing your info, tracking skills)",
            "planner_agent": "Study plan generation (personalized schedules)",
            "interviewer_agent": "Mock interview simulation (realistic practice)",
            "evaluator_agent": "Answer evaluation (scoring and feedback)",
            "strategy_agent": "Company-specific strategy (tailored preparation)",
        }
        desc = agent_descriptions.get(agent_name, "this feature")

        return {
            **state,
            "agent_response": (
                f"Got it — you want help with {desc}. "
                f"This agent is coming in the next sprint. "
                f"For now, I can help you with general chat. "
                f"What else would you like to work on?"
            ),
            "current_agent": agent_name,
            "is_complete": True,
        }

    return placeholder


def route_after_router(state: AgentState) -> str:
    """Conditional edge function: Router → specialist agent.

    This is the key routing logic. Based on the Router's classification,
    we send the user to the correct specialist agent.

    Interview talking point: 'Conditional edges in LangGraph — not hardcoded
    if/else in the agent. The routing logic is a pure function of state,
    which makes it independently testable. I can unit test every routing
    path without running any LLM.'
    """
    agent = state.get("current_agent", "general_chat")

    # Safety: if we've exceeded max steps, terminate
    if state.get("step_count", 0) > 5:
        return "general_chat"

    # Map agent names to graph node names
    valid_nodes = {
        "profile_agent",
        "planner_agent",
        "interviewer_agent",
        "evaluator_agent",
        "strategy_agent",
        "general_chat",
        "clarify",
    }

    if agent in valid_nodes:
        return agent

    return "general_chat"


def should_end(state: AgentState) -> str:
    """Check if the graph should terminate or continue.

    After each specialist agent runs, we check:
    - is_complete=True → end (return response to user)
    - is_complete=False → route back to router (multi-turn within a request)
    """
    if state.get("is_complete", False):
        return "end"
    return "continue"


def build_graph(llm: LLMGateway | None = None) -> StateGraph:
    """Build and compile the full AgentCoach graph.

    Architecture:
    1. 'router' node classifies intent
    2. Conditional edge routes to specialist agent
    3. Specialist agent processes and sets is_complete=True
    4. Graph terminates and returns response

    Returns compiled graph ready for .invoke() or .stream().

    Interview talking point: 'The graph is built by a factory function that
    takes the LLM gateway as a dependency. This makes testing easy — I can
    inject a mock LLM and test the full graph flow without hitting Azure
    OpenAI. Dependency injection at the graph level.'
    """
    if llm is None:
        llm = LLMGateway()

    # Initialize agents
    router = RouterAgent(llm=llm)

    # Build graph
    graph = StateGraph(AgentState)

    # ── Add nodes ──
    graph.add_node("router", router.invoke)
    graph.add_node("general_chat", create_general_chat_node(llm))
    graph.add_node("clarify", create_clarify_node(llm))

    # Placeholder nodes for agents we'll build in Epics 2-5
    graph.add_node("profile_agent", create_placeholder_node("profile_agent"))
    graph.add_node("planner_agent", create_placeholder_node("planner_agent"))
    graph.add_node("interviewer_agent", create_placeholder_node("interviewer_agent"))
    graph.add_node("evaluator_agent", create_placeholder_node("evaluator_agent"))
    graph.add_node("strategy_agent", create_placeholder_node("strategy_agent"))

    # ── Add edges ──

    # Entry point: always start at router
    graph.set_entry_point("router")

    # Router → specialist agent (conditional)
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "profile_agent": "profile_agent",
            "planner_agent": "planner_agent",
            "interviewer_agent": "interviewer_agent",
            "evaluator_agent": "evaluator_agent",
            "strategy_agent": "strategy_agent",
            "general_chat": "general_chat",
            "clarify": "clarify",
        },
    )

    # All specialist agents → END
    # (In future epics, some agents may route back to router for multi-step)
    graph.add_edge("general_chat", END)
    graph.add_edge("clarify", END)
    graph.add_edge("profile_agent", END)
    graph.add_edge("planner_agent", END)
    graph.add_edge("interviewer_agent", END)
    graph.add_edge("evaluator_agent", END)
    graph.add_edge("strategy_agent", END)

    # Compile
    compiled = graph.compile()
    return compiled


def run_agent(user_input: str, state: AgentState | None = None) -> AgentState:
    """Convenience function: run a single user message through the graph.

    Usage:
        result = run_agent("Hi, I want to prepare for Google interviews")
        print(result["agent_response"])

    Interview talking point: 'Single entry point for the entire system.
    Pass a message, get a response. The graph handles routing, agent
    selection, state management — the caller doesn't need to know about
    the multi-agent internals.'
    """
    if state is None:
        state = {
            "messages": [],
            "current_input": user_input,
            "current_mode": "chat",
            "current_agent": "",
            "route_confidence": 0.0,
            "agent_response": "",
            "step_count": 0,
            "is_complete": False,
            "error": None,
        }
    else:
        state = {
            **state,
            "current_input": user_input,
            "is_complete": False,
            "error": None,
        }

    graph = build_graph()
    result = graph.invoke(state)

    # Append to message history
    messages = result.get("messages", [])
    messages.append({"role": "user", "content": user_input})
    if result.get("agent_response"):
        messages.append({"role": "assistant", "content": result["agent_response"]})
    result["messages"] = messages

    return result