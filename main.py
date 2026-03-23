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
    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")

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

            # Send immediate confirmation so frontend knows connection is alive
            yield f"data: {json.dumps({'type': 'log', 'message': 'Connected — 7 agents starting…'})}\n\n"
            await asyncio.sleep(0.1)

            last_ping = asyncio.get_event_loop().time()

            # Stream progress as each agent node completes
            async for step in flex_graph.astream(initial_state):
                node_name = list(step.keys())[0] if step else None
                node_state = step.get(node_name, {}) if node_name else {}

                # Send heartbeat ping every 10s to keep connection alive
                now = asyncio.get_event_loop().time()
                if now - last_ping > 10:
                    yield f"data: {json.dumps({'type': 'ping'})}\n\n"
                    last_ping = now

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
    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")

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
async def generate_video_script(request: VideoScriptRequest):
    from agents import call_gemini, parse_json

    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")

    try:
        system = """You are the Video Agent in Flex AI. Write a 60-second tutorial video script specific to the project and step.
Respond ONLY with valid JSON, no markdown:
{
  "title": "short video title max 6 words",
  "narration": "spoken script 80-100 words, reference exact project name and tools, friendly and encouraging",
  "captions": [{"t": 0, "text": "caption"}, {"t": 6, "text": "next"}],
  "code_lines": ["line1", "line2", "line3", "line4", "line5"]
}
Include 8-10 caption objects. code_lines: key real code lines max 8."""

        user = f"""Problem: {request.problem_scope}
Project: {request.project_title} ({request.project_type})
Stack: {', '.join(request.stack)}
Difficulty: {request.difficulty}
Step type: {request.step_type}
Step: "{request.step_title}"
Description: {request.step_desc}

Write the 60-second video script."""

        result = await call_gemini(system, user, 800)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Tutorial step generation — called when user picks a project ───────────────
@app.post("/api/tutorial/steps")
async def generate_tutorial_steps(request: dict):
    from agents import call_gemini

    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")

    try:
        project = request.get("project", {})
        problem = request.get("problem", "")
        starter_code = project.get("starter_code") or {}
        tools = project.get("tools") or []
        tools_str = ", ".join([t.get("name","") + " (" + t.get("url","") + ")" for t in tools])

        system = """You are the Tutorial Agent in Flex AI generating detailed step-by-step instructions.
Respond ONLY with valid JSON, no markdown:
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
        {"t": "wire", "label": "component wiring"}
      ],
      "errors": ["real error 1", "real error 2"],
      "verify": "what user should see to confirm step worked"
    }
  ]
}
Rules: 6-8 steps, real runnable code only, wire type only on hardware steps."""

        user = f"""Problem: {problem}
Project: {project.get('title','')} ({project.get('type','')})
Stack: {', '.join(project.get('stack',[]))}
Difficulty: {project.get('difficulty','')}
Description: {project.get('description','')}

Starter code ({starter_code.get('filename','main.py')}):
Install: {starter_code.get('install','')}
{starter_code.get('code','')}

Tools: {tools_str}

Write the complete tutorial with real code."""

        result = await call_gemini(system, user, 4000)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
