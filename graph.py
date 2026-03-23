"""
Flex AI — LangGraph agent graph
Defines the graph topology: sequential agents feeding into parallel agents,
then synthesis, then video agent queuing.
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, Any
import asyncio
from agents import (
    planner_agent,
    stack_scout_agent,
    budget_bot_agent,
    tutorial_agent,
    code_agent,
    tools_sourcer_agent,
    video_agent,
)


# ── State schema ──────────────────────────────────────────────────────────────
class FlexAIState(TypedDict):
    # Inputs
    problem: str
    budget: int

    # Agent outputs
    planner: Optional[dict]
    stack_scout: Optional[dict]
    budget_bot: Optional[dict]
    tutorial: Optional[dict]
    code_agent: Optional[dict]
    tools_sourcer: Optional[dict]
    video_agent: Optional[dict]

    # System
    log: list
    error: Optional[str]


# ── Parallel node: runs Tutorial + Code + Tools simultaneously ────────────────
async def parallel_agents(state: FlexAIState) -> FlexAIState:
    """
    Runs Tutorial Agent, Code Agent and Tools Sourcer in parallel.
    All three receive the full upstream context simultaneously.
    Results are merged back into state.
    """
    results = await asyncio.gather(
        tutorial_agent(state),
        code_agent(state),
        tools_sourcer_agent(state),
        return_exceptions=True
    )

    merged = {**state}
    for result in results:
        if isinstance(result, Exception):
            merged["log"] = merged.get("log", []) + [f"⚠️ Agent error: {str(result)}"]
        elif isinstance(result, dict):
            # Merge each agent's output keys
            for key in ["tutorial", "code_agent", "tools_sourcer", "log"]:
                if key in result:
                    if key == "log":
                        merged["log"] = merged.get("log", []) + result.get("log", [])
                    else:
                        merged[key] = result[key]

    return merged


# ── Synthesis node: merges all agent outputs into final project list ──────────
async def synthesise(state: FlexAIState) -> FlexAIState:
    """
    Combines outputs from all agents into a clean final projects list
    that the frontend can consume directly.
    """
    stack_scout = state.get("stack_scout", {})
    tutorial = state.get("tutorial", {})
    budget_bot = state.get("budget_bot", {})
    code_agent = state.get("code_agent", {})
    tools_sourcer = state.get("tools_sourcer", {})

    solutions = stack_scout.get("solutions", [])
    tut_solutions = {s.get("title"): s for s in tutorial.get("solutions", [])}
    budget_solutions = {s.get("title"): s for s in budget_bot.get("solutions", [])}
    code_snippets = {s.get("title"): s for s in code_agent.get("snippets", [])}
    tool_solutions = {s.get("title"): s for s in tools_sourcer.get("solutions", [])}

    projects = []
    for sol in solutions:
        title = sol.get("title", "")
        tut = tut_solutions.get(title, {})
        budget = budget_solutions.get(title, {})
        code = code_snippets.get(title, {})
        tools = tool_solutions.get(title, {})

        projects.append({
            # Identity
            "title": title,
            "tagline": tut.get("tagline", sol.get("justification", "")),
            "type": sol.get("type", "software"),
            "difficulty": sol.get("difficulty", "intermediate"),
            "tags": sol.get("tags", []),

            # Description from Tutorial Agent
            "description": tut.get("description", ""),
            "best_for": tut.get("best_for", ""),
            "estimated_total_time": tut.get("estimated_total_time", ""),

            # Stack from Stack Scout
            "stack": sol.get("stack", []),
            "justification": sol.get("justification", ""),
            "prerequisites": sol.get("prerequisites", []),
            "gotchas": sol.get("gotchas", []),

            # Cost from Budget Bot
            "estimated_cost": budget.get("estimated_cost", ""),
            "cost_breakdown": budget.get("breakdown", []),
            "total_monthly": budget.get("total_monthly", ""),
            "free_alternative": budget.get("free_alternative"),
            "within_budget": budget.get("within_budget", True),

            # Phases from Tutorial Agent
            "phases": tut.get("phases", []),

            # Starter code from Code Agent
            "starter_code": {
                "filename": code.get("filename", "main.py"),
                "lang": code.get("lang", "python"),
                "install": code.get("install", ""),
                "code": code.get("code", ""),
                "what_it_does": code.get("what_it_does", ""),
            } if code else None,

            # Tools from Tools Sourcer
            "tools": tools.get("tools", []),
        })

    return {
        **state,
        "projects": projects,
        "log": state.get("log", []) + [f"✓ Synthesis complete — {len(projects)} solutions ready"]
    }


# ── Build the graph ───────────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(dict)

    # Add nodes
    graph.add_node("planner", planner_agent)
    graph.add_node("stack_scout", stack_scout_agent)
    graph.add_node("budget_bot", budget_bot_agent)
    graph.add_node("parallel_agents", parallel_agents)
    graph.add_node("video_agent_node", video_agent)
    graph.add_node("synthesise", synthesise)

    # Sequential edges
    graph.add_edge("planner", "stack_scout")
    graph.add_edge("stack_scout", "budget_bot")
    graph.add_edge("budget_bot", "parallel_agents")
    graph.add_edge("parallel_agents", "video_agent_node")
    graph.add_edge("video_agent_node", "synthesise")
    graph.add_edge("synthesise", END)

    # Entry point
    graph.set_entry_point("planner")

    return graph.compile()


# Compiled graph instance
flex_graph = build_graph()
