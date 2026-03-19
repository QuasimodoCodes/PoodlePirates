# Session Handoff — Tripletex AI Accounting Agent

## What this project is
A competition agent for https://app.ainm.no/submit/tripletex
The platform sends POST /solve with a task prompt + Tripletex sandbox credentials.
The agent uses Google Gemini (free) to interpret the task and call Tripletex APIs.

## Current status
- Step 1 ✅ Project scaffolding, folder structure, conda env `tripletex_agent`
- Step 2 ✅ FastAPI /solve endpoint + dynamic TripletexClient (per-request credentials)
- Step 3 ✅ Gemini tool-calling agent loop (4 tools: GET/POST/PUT/DELETE)
- Step 4 ⏳ NEXT: Get Gemini API key, test end-to-end, then deploy publicly

## How to resume
1. Open VS Code in: C:\Users\herma\PoodlePirates\Tripletex_AI_Accounting_Agent
2. conda activate tripletex_agent
3. Add Gemini key to .env:  GEMINI_API_KEY=AIza...
   Get free key at: https://aistudio.google.com/app/apikey
4. python main.py   ← starts the server on port 8000
5. Run test: python tests/test_solve_local.py
   (set env vars TRIPLETEX_PROXY_URL and TRIPLETEX_SESSION_TOKEN first)

## Key files
- src/app.py          — FastAPI server, POST /solve endpoint
- src/agent/runner.py — Gemini tool-calling loop
- src/agent/prompt.py — System prompt with Tripletex instructions
- src/tripletex/client.py — HTTP client (takes proxy_url + session_token per request)
- src/config.py       — Reads GEMINI_API_KEY from .env
- main.py             — Run this to start the server
- tests/test_solve_local.py — Local end-to-end test

## Remaining steps (see plan.json for full detail)
4. Tripletex tools registry (more specific tools for common tasks)
5. File handling (PDF/image attachments in /solve requests)
6. Error handling + timeout guard
7. Efficiency optimization (cache lookups within a request)
8. Deployment to public HTTPS (Railway or Render — free tier)
9. End-to-end test with real sandbox credentials
