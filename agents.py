"""
Flex AI — Agent definitions
Each agent is a pure async function that receives the shared state
and returns an updated state. LangGraph calls them as graph nodes.
"""

import json
import re
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

# ── Shared model instance ─────────────────────────────────────────────────────
def get_model():
    return ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=1500)


def parse_json(text: str) -> dict:
    """Safely extract JSON from a Claude response."""
    text = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        # Try to find first { ... } block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return {}


# ── AGENT 0: PLANNER ─────────────────────────────────────────────────────────
async def planner_agent(state: dict) -> dict:
    """
    Receives the raw problem and produces a structured brief.
    Defines scope, success criteria, constraints and 3 high-level approaches.
    """
    model = get_model()
    problem = state["problem"]
    budget = state["budget"]

    system = """You are the Planner agent in Flex AI — a platform that helps anyone solve a problem using technology, regardless of their background.

Your job: receive a raw problem statement and produce a precise, structured brief that every downstream agent will use as their source of truth.

Respond ONLY with valid JSON:
{
  "scope": "one sentence — the exact problem being solved",
  "success_criteria": ["criterion 1", "criterion 2", "criterion 3"],
  "constraints": ["constraint 1", "constraint 2"],
  "problem_type": "software | hardware | ai | hybrid",
  "complexity": "low | medium | high",
  "approaches": [
    "approach 1 — one sentence describing a software solution",
    "approach 2 — one sentence describing a hardware or physical solution",
    "approach 3 — one sentence describing an AI-powered solution"
  ],
  "target_user": "brief description of who this person likely is"
}"""

    human = f"Problem: {problem}\nBudget: ${budget}\n\nAnalyse this problem and produce a structured brief."

    response = model.invoke([SystemMessage(content=system), HumanMessage(content=human)])
    result = parse_json(response.content)

    return {**state, "planner": result, "log": state.get("log", []) + ["🧠 Planner: brief defined — " + result.get("scope", "scope undefined")]}


# ── AGENT 1: STACK SCOUT ─────────────────────────────────────────────────────
async def stack_scout_agent(state: dict) -> dict:
    """
    Receives the Planner brief and identifies the optimal tech stack
    for each of the 3 approaches. Justifies every choice.
    """
    model = get_model()
    planner = state.get("planner", {})

    system = """You are the Stack Scout agent in Flex AI.

Your job: receive a problem brief and identify the optimal technology stack for 3 different solution approaches. You are an expert across software, hardware, AI and embedded systems.

For each solution you must:
- Pick the exact tools (not categories — specific libraries, boards, services)
- Justify why this stack over alternatives
- Flag any gotchas or prerequisites

Respond ONLY with valid JSON:
{
  "solutions": [
    {
      "title": "ProductStyleName",
      "type": "software | hardware | ai",
      "stack": ["exact tool 1", "exact tool 2", "exact tool 3"],
      "justification": "why this stack specifically",
      "difficulty": "beginner | intermediate | advanced",
      "tags": ["tag1", "tag2", "tag3"],
      "prerequisites": ["prereq 1", "prereq 2"],
      "gotchas": ["watch out for this"]
    }
  ]
}
Exactly 3 solutions. Product-style titles (e.g. ExpenseBot, VisionDoor, DocMind)."""

    human = f"""Problem brief from Planner:
Scope: {planner.get('scope', '')}
Problem type: {planner.get('problem_type', '')}
Complexity: {planner.get('complexity', '')}
Approaches: {json.dumps(planner.get('approaches', []))}
Budget: ${state['budget']}

Identify the best tech stack for 3 different solutions to this problem."""

    response = model.invoke([SystemMessage(content=system), HumanMessage(content=human)])
    result = parse_json(response.content)

    titles = [s.get("title", "") for s in result.get("solutions", [])]
    return {**state, "stack_scout": result, "log": state.get("log", []) + [f"🔍 Stack Scout: {', '.join(titles)} — stacks identified"]}


# ── AGENT 2: BUDGET BOT ───────────────────────────────────────────────────────
async def budget_bot_agent(state: dict) -> dict:
    """
    Receives the Stack Scout's recommendations and produces a real
    per-item cost breakdown for each solution. Flags free alternatives.
    """
    model = get_model()
    stack_scout = state.get("stack_scout", {})
    budget = state["budget"]

    solutions_summary = "\n".join([
        f"{i+1}. {s.get('title', '')} ({s.get('type', '')}): {', '.join(s.get('stack', []))}"
        for i, s in enumerate(stack_scout.get("solutions", []))
    ])

    system = """You are the Budget Bot agent in Flex AI.

Your job: receive tech stacks and calculate realistic, specific costs for each solution. You know current pricing for cloud services, hardware components, APIs and developer tools.

Be specific — not "hosting ~$5-20" but "Railway hobby plan: $5/mo" or "free on Vercel hobby tier".

Respond ONLY with valid JSON:
{
  "solutions": [
    {
      "title": "...",
      "estimated_cost": "Free | $X one-time | $X/mo",
      "breakdown": [
        {"item": "exact item name", "cost": "Free | $X", "note": "why needed / free alternative"}
      ],
      "total_one_time": "$X or Free",
      "total_monthly": "$X/mo or Free",
      "free_alternative": "description of fully free version if exists | null",
      "within_budget": true
    }
  ]
}"""

    human = f"""Budget available: ${budget}

Stacks to cost:
{solutions_summary}

Calculate realistic costs. Prefer free options. Flag anything that might surprise someone."""

    response = model.invoke([SystemMessage(content=system), HumanMessage(content=human)])
    result = parse_json(response.content)

    cost_summary = " | ".join([
        f"{s.get('title', '')}: {s.get('estimated_cost', '?')}"
        for s in result.get("solutions", [])
    ])
    return {**state, "budget_bot": result, "log": state.get("log", []) + [f"💰 Budget Bot: {cost_summary}"]}


# ── AGENT 3: TUTORIAL AGENT ──────────────────────────────────────────────────
async def tutorial_agent(state: dict) -> dict:
    """
    Receives all upstream context and writes the full solution descriptions
    and build phases. Knows exact stack, exact budget, exact constraints.
    """
    model = get_model()
    planner = state.get("planner", {})
    stack_scout = state.get("stack_scout", {})
    budget_bot = state.get("budget_bot", {})

    solutions_detail = "\n".join([
        f"{i+1}. {s.get('title', '')} — Stack: {', '.join(s.get('stack', []))} — Difficulty: {s.get('difficulty', '')}"
        for i, s in enumerate(stack_scout.get("solutions", []))
    ])

    system = """You are the Tutorial Agent in Flex AI.

Your job: receive the full context from all upstream agents and write a precise, practical description and build roadmap for each solution. You write for people with NO technical background — clear, encouraging, jargon-free.

Respond ONLY with valid JSON:
{
  "solutions": [
    {
      "title": "...",
      "description": "2 sentences explaining exactly how this solves the user's problem. Mention the key technology once.",
      "tagline": "one punchy line — what it does",
      "phases": [
        {
          "phase": "Phase title",
          "duration": "X days",
          "steps": ["step 1", "step 2", "step 3"]
        }
      ],
      "estimated_total_time": "X hours",
      "best_for": "who this solution suits best"
    }
  ]
}"""

    human = f"""User problem: {planner.get('scope', state['problem'])}
Success criteria: {json.dumps(planner.get('success_criteria', []))}
Target user: {planner.get('target_user', 'general user')}
Budget: ${state['budget']}

Solutions and stacks:
{solutions_detail}

Write a description and 2-3 build phases per solution."""

    response = model.invoke([SystemMessage(content=system), HumanMessage(content=human)])
    result = parse_json(response.content)

    return {**state, "tutorial": result, "log": state.get("log", []) + ["📋 Tutorial Agent: build phases written"]}


# ── AGENT 4: CODE AGENT ───────────────────────────────────────────────────────
async def code_agent(state: dict) -> dict:
    """
    Writes a real, runnable starter code snippet for each solution.
    Not pseudocode — actual working code specific to the chosen stack.
    """
    model = get_model()
    planner = state.get("planner", {})
    stack_scout = state.get("stack_scout", {})

    solutions_detail = "\n".join([
        f"{i+1}. {s.get('title', '')} — Stack: {', '.join(s.get('stack', []))} — Type: {s.get('type', '')}"
        for i, s in enumerate(stack_scout.get("solutions", []))
    ])

    system = """You are the Code Agent in Flex AI.

Your job: write a real, runnable, working starter code snippet for each solution. This is the first file a user would create — not a full app, but enough to prove the concept works.

Rules:
- Real code only. No pseudocode, no placeholders like "# add your logic here"
- Use actual package names, actual APIs, actual syntax
- 10-20 lines per snippet — enough to be meaningful, not overwhelming
- Include a comment at the top with the install command

Respond ONLY with valid JSON:
{
  "snippets": [
    {
      "title": "...",
      "filename": "main.py or index.js etc",
      "lang": "python | javascript | typescript | bash",
      "install": "pip install X Y Z or npm install X Y Z",
      "code": "the actual working code",
      "what_it_does": "one sentence"
    }
  ]
}"""

    human = f"""Problem being solved: {planner.get('scope', state['problem'])}

Solutions:
{solutions_detail}

Write a real starter code snippet for each solution. Specific to the exact stack listed."""

    response = model.invoke([SystemMessage(content=system), HumanMessage(content=human)])
    result = parse_json(response.content)

    return {**state, "code_agent": result, "log": state.get("log", []) + ["🤖 Code Agent: starter snippets generated"]}


# ── AGENT 5: TOOLS SOURCER ────────────────────────────────────────────────────
async def tools_sourcer_agent(state: dict) -> dict:
    """
    Finds the most useful real documentation, tools and resources
    for each solution. Returns actual URLs.
    """
    model = get_model()
    stack_scout = state.get("stack_scout", {})

    solutions_summary = "\n".join([
        f"{i+1}. {s.get('title', '')} — Stack: {', '.join(s.get('stack', []))}"
        for i, s in enumerate(stack_scout.get("solutions", []))
    ])

    system = """You are the Tools Sourcer agent in Flex AI.

Your job: find the most useful documentation, tutorials and tools for each solution. Real URLs only — no invented links.

Respond ONLY with valid JSON:
{
  "solutions": [
    {
      "title": "...",
      "tools": [
        {
          "name": "tool name",
          "url": "https://real-url.com",
          "free": true,
          "category": "docs | tutorial | tool | community",
          "note": "why this is useful — one sentence"
        }
      ]
    }
  ]
}
3-5 tools per solution. Prioritise official docs, then quality tutorials, then community."""

    human = f"""Solutions and stacks:
{solutions_summary}

Find the most useful real resources for each solution."""

    response = model.invoke([SystemMessage(content=system), HumanMessage(content=human)])
    result = parse_json(response.content)

    return {**state, "tools_sourcer": result, "log": state.get("log", []) + ["📦 Tools Sourcer: resources found"]}


# ── AGENT 6: VIDEO AGENT ─────────────────────────────────────────────────────
async def video_agent(state: dict) -> dict:
    """
    Queues video script generation for each tutorial step.
    In Phase 1 this prepares the script context.
    Actual rendering happens per-step when the user enters the tutorial.
    """
    stack_scout = state.get("stack_scout", {})
    planner = state.get("planner", {})

    # In Phase 1: queue metadata per solution — scripts generated on demand per step
    queued = []
    for sol in stack_scout.get("solutions", []):
        queued.append({
            "title": sol.get("title", ""),
            "type": sol.get("type", ""),
            "stack": sol.get("stack", []),
            "problem_scope": planner.get("scope", state["problem"]),
            "status": "queued"
        })

    return {
        **state,
        "video_agent": {"queued": queued, "status": "ready"},
        "log": state.get("log", []) + [f"🎬 Video Agent: {len(queued)} video queues ready — scripts generate per step"]
    }
