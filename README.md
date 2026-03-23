# Flex AI — Backend

True multi-agent AI system built with Python, FastAPI and LangGraph.

## Architecture

```
User problem
    ↓
🧠 Planner          — defines scope, criteria, constraints
    ↓
🔍 Stack Scout      — picks exact tech stack per approach
    ↓
💰 Budget Bot       — real per-item cost breakdown
    ↓ ↓ ↓ (parallel)
📋 Tutorial Agent   — writes phases and descriptions
🤖 Code Agent       — writes real runnable starter code
📦 Tools Sourcer    — finds real docs and tool URLs
    ↓
🎬 Video Agent      — queues per-step video scripts
    ↓
Synthesis           — merges all agent outputs
    ↓
Final projects JSON → frontend
```

## Local setup

```bash
# 1. Clone and enter directory
cd flexai-backend

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# 5. Run locally
uvicorn main:app --reload --port 8000
```

API will be running at http://localhost:8000

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | Health check |
| GET | /health | Health check |
| POST | /api/projects | Run full pipeline (non-streaming) |
| POST | /api/projects/stream | Run full pipeline (SSE streaming) |
| POST | /api/tutorial/steps | Generate tutorial steps for a project |
| POST | /api/video/script | Generate video script for a step |

### POST /api/projects

```json
{
  "problem": "my plants keep dying because I forget to water them",
  "budget": 50
}
```

Returns:
```json
{
  "projects": [...],
  "log": [...],
  "planner": {...}
}
```

### POST /api/projects/stream

Same request body. Returns SSE stream:

```
data: {"type": "log", "message": "🧠 Planner: brief defined..."}
data: {"type": "log", "message": "🔍 Stack Scout: ..."}
data: {"type": "result", "projects": [...]}
data: {"type": "done"}
```

### POST /api/tutorial/steps

```json
{
  "problem": "my plants keep dying",
  "project": { ...full project object from /api/projects... }
}
```

### POST /api/video/script

```json
{
  "project_title": "PlantMind",
  "project_type": "hardware",
  "stack": ["UNIHIKER K10", "Soil sensor", "MicroPython"],
  "problem_scope": "Plants dying because user forgets to water",
  "step_title": "Wire the soil sensor",
  "step_desc": "Connect the moisture sensor to GPIO pin P0",
  "step_type": "hardware",
  "difficulty": "beginner"
}
```

## Deploy to Railway

1. Push this folder to a GitHub repository
2. Go to [railway.app](https://railway.app) and create a new project
3. Connect your GitHub repo
4. Add environment variable: `ANTHROPIC_API_KEY = your_key_here`
5. Railway auto-detects the Procfile and deploys

Your backend will be live at `https://your-app.railway.app`

## Connect the frontend

In `flexai.html`, set the backend URL:

```javascript
const BACKEND_URL = 'https://your-app.railway.app';
```

Then replace the `runSwarm()` function to call `/api/projects/stream`
and stream agent progress events to the UI.

## File structure

```
flexai-backend/
├── main.py          — FastAPI app, all endpoints
├── graph.py         — LangGraph agent graph definition
├── agents.py        — All 7 agent functions
├── requirements.txt — Python dependencies
├── Procfile         — Railway start command
├── railway.json     — Railway config
├── .env.example     — Environment variables template
└── .gitignore
```
