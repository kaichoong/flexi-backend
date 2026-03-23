"""
Flex AI — FastAPI backend
Exposes the multi-agent graph as a REST API with SSE streaming
so the frontend can show real-time agent progress.
"""

import os
import json
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from graph import flex_graph

load_dotenv()

app = FastAPI(title="Flex AI Backend", version="1.0.0")

# Allow all origins in dev — tighten for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────
class ProjectRequest(BaseModel):
    problem: str
    budget: int = 50


class VideoScriptRequest(BaseModel):
    project_title: str
    project_type: str
    stack: list[str]
    problem_scope: str
    step_title: str
    step_desc: str
    step_type: str
    difficulty: str


# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "Flex AI backend running", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Main agent pipeline — streaming SSE ──────────────────────────────────────
@app.post("/api/projects/stream")
async def stream_projects(request: ProjectRequest):
    """
    Runs the full 7-agent pipeline and streams progress events
    back to the frontend via Server-Sent Events.

    Event types:
    - log      → agent progress message
    - result   → final projects JSON
    - error    → something went wrong
    - done     → stream complete
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")

    async def event_stream():
        try:
            initial_state = {
                "problem": request.problem,
                "budget": request.budget,
                "planner": None,
                "stack_scout": None,
                "budget_bot": None,
                "tutorial": None,
                "code_agent": None,
                "tools_sourcer": None,
                "video_agent": None,
                "log": [],
                "error": None,
            }

            # Stream progress as each agent node completes
            async for step in flex_graph.astream(initial_state):
                node_name = list(step.keys())[0] if step else None
                node_state = step.get(node_name, {}) if node_name else {}

                # Emit any new log entries
                for entry in node_state.get("log", []):
                    yield f"data: {json.dumps({'type': 'log', 'message': entry})}\n\n"
                    await asyncio.sleep(0.05)

                # Emit final result when synthesis completes
                if node_name == "synthesise" and "projects" in node_state:
                    yield f"data: {json.dumps({'type': 'result', 'projects': node_state['projects']})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
    "Access-Control-Allow-Origin": "*",
        }
    )


# ── Non-streaming fallback ────────────────────────────────────────────────────
@app.post("/api/projects")
async def get_projects(request: ProjectRequest):
    """
    Non-streaming version — waits for the full pipeline then returns results.
    Use this as fallback if SSE isn't supported.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")

    try:
        initial_state = {
            "problem": request.problem,
            "budget": request.budget,
            "planner": None,
            "stack_scout": None,
            "budget_bot": None,
            "tutorial": None,
            "code_agent": None,
            "tools_sourcer": None,
            "video_agent": None,
            "log": [],
            "error": None,
        }

        final_state = await flex_graph.ainvoke(initial_state)

        return {
            "projects": final_state.get("projects", []),
            "log": final_state.get("log", []),
            "planner": final_state.get("planner", {}),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Video script generation — called per tutorial step ───────────────────────
@app.post("/api/video/script")
yield f"data: {json.dumps({'type': 'log', 'message': 'Connected - 7 agents starting...'})}\n\n"
await asyncio.sleep(0.1)
async def generate_video_script(request: VideoScriptRequest):
    """
    Generates a video script for a single tutorial step.
    Called by the frontend Video Agent per step when the user enters the tutorial.
    """
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import SystemMessage, HumanMessage
    from agents import parse_json

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")

    try:
        model = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=800)

        system = """You are the Video Agent in Flex AI. Write a tight 60-second tutorial video script that is hyper-specific to the exact project, stack and step.

Respond ONLY with valid JSON:
{
  "title": "short video title max 6 words",
  "narration": "spoken script 80-100 words — reference the exact project name, exact tools, exact commands. Friendly and encouraging.",
  "captions": [
    {"t": 0, "text": "caption text"},
    {"t": 6, "text": "next caption"}
  ],
  "code_lines": ["line1", "line2", "line3", "line4", "line5"]
}
Include 8-10 caption objects. code_lines: key real code lines for this step (max 8)."""

        human = f"""User problem: {request.problem_scope}
Project: {request.project_title} ({request.project_type})
Stack: {', '.join(request.stack)}
Difficulty: {request.difficulty}
Step type: {request.step_type}
Step: "{request.step_title}"
Description: {request.step_desc}

Write the 60-second video script."""

        response = model.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        result = parse_json(response.content)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Tutorial step generation — called when user picks a project ───────────────
@app.post("/api/tutorial/steps")
async def generate_tutorial_steps(request: dict):
    """
    Generates the full step-by-step tutorial for a chosen project.
    Receives the complete project object (from /api/projects) and
    returns 6-8 detailed tutorial steps.
    """
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import SystemMessage, HumanMessage
    from agents import parse_json

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")

    try:
        model = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=5000)
        project = request.get("project", {})
        problem = request.get("problem", "")

        starter_code = project.get("starter_code", {})
        tools = project.get("tools", [])
        tools_str = ", ".join([t.get("name", "") + " (" + t.get("url", "") + ")" for t in tools])

        system = """You are the Tutorial Agent in Flex AI generating detailed step-by-step instructions.

Respond ONLY with valid JSON:
{
  "steps": [
    {
      "id": 0,
      "type": "concept | setup | hardware | code | terminal | config | deploy | test",
      "title": "...",
      "desc": "...",
      "dur": "X min",
      "content": [
        {"t": "prose", "v": "HTML with <strong> tags ok"},
        {"t": "callout", "v": "info | warn | success", "text": "..."},
        {"t": "code", "lang": "python | bash | javascript", "editable": true, "code": "REAL code", "expected": "expected output"},
        {"t": "wire", "label": "component wiring description"}
      ],
      "errors": ["specific real error 1", "specific real error 2"],
      "verify": "exactly what the user should see to confirm this step worked"
    }
  ]
}

Rules:
- 6-8 steps
- Real, specific, runnable code — not pseudocode
- Wire type only on hardware steps
- errors must be real errors someone would actually hit
- Use the starter code snippet provided in the relevant step"""

        human = f"""User problem: {problem}
Project: {project.get('title', '')} ({project.get('type', '')})
Stack: {', '.join(project.get('stack', []))}
Difficulty: {project.get('difficulty', '')}
Description: {project.get('description', '')}
Prerequisites: {', '.join(project.get('prerequisites', []))}
Gotchas: {', '.join(project.get('gotchas', []))}

Starter code:
Filename: {starter_code.get('filename', 'main.py')}
Language: {starter_code.get('lang', 'python')}
Install: {starter_code.get('install', '')}
Code:
{starter_code.get('code', '')}

Key tools and docs: {tools_str}

Write the complete tutorial. Real code only."""

        response = model.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        result = parse_json(response.content)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
