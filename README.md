# SHL Assessment Recommendation Agent

A conversational FastAPI service that recommends SHL individual assessments through dialogue.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Readiness check → `{"status": "ok"}` |
| POST | `/chat` | Multi-turn assessment recommendation |

## Request / Response

```json
POST /chat
{
  "messages": [
    {"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
    {"role": "assistant", "content": "Sure. What seniority level?"},
    {"role": "user", "content": "Mid-level, around 4 years"}
  ]
}

→ 200 OK
{
  "reply": "Here are 5 assessments for a mid-level Java developer.",
  "recommendations": [
    {"name": "Core Java (Advanced Level) (New)", "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "Occupational Personality Questionnaire OPQ32r", "url": "https://www.shl.com/...", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

- `recommendations` is `[]` while clarifying or refusing.
- `end_of_conversation` is `true` only when the agent considers the task complete.
- Maximum 8 turns per conversation (stateless — full history sent each call).

## Local Development

```bash
pip install -r requirements.txt
export AZURE_OPENAI_API_KEY=your-azure-key
export AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE.openai.azure.com/
export AZURE_OPENAI_API_VERSION=2024-02-01
export AZURE_OPENAI_DEPLOYMENT=gpt-4o
uvicorn main:app --reload
```

## Deploy to Render

1. Push this directory to a GitHub repo.
2. Create a new **Web Service** on [render.com](https://render.com), connect the repo.
3. Set environment variable `ANTHROPIC_API_KEY` in Render's dashboard.
4. Render auto-detects `render.yaml` — build and start commands are pre-configured.
5. First health check allows up to 2 minutes for cold start.

## Deploy with Docker

```bash
docker build -t shl-agent .
docker run \
  -e AZURE_OPENAI_API_KEY=your-azure-key \
  -e AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE.openai.azure.com/ \
  -e AZURE_OPENAI_API_VERSION=2024-02-01 \
  -e AZURE_OPENAI_DEPLOYMENT=gpt-4o \
  -p 8000:8000 shl-agent
```

## Testing

```bash
# Offline unit + integration tests (no API key needed)
python test_agent.py --url http://localhost:8000

# Or against a deployed endpoint
python test_agent.py --url https://your-service.onrender.com
```

## Architecture

- **Retrieval**: TF-IDF over 377 catalogue items + keyword boosts for anchor products.
  Each query retrieves top-25 candidates injected into the system prompt (~1,200 tokens).
- **LLM**: Claude Haiku 4.5 for speed; swap to Sonnet in `main.py` for higher quality.
- **Safety**: Pre-LLM keyword filter catches injection/off-topic. Post-LLM URL validator
  drops any hallucinated items not in the catalogue.
- **Stateless**: No per-session storage. Full history sent on every request.
