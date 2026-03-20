"""
Local integration test for the /solve endpoint.
Simulates a request from the competition platform.

Usage:
    conda activate tripletex_agent
    python tests/test_solve_local.py

Requirements:
    - Server running: python main.py (in another terminal)
    - .env has a valid ANTHROPIC_API_KEY
    - Edit PROXY_URL and SESSION_TOKEN below with real sandbox credentials
      (or set them as env vars)
"""
import os
import httpx

# ── Fill these in with your sandbox credentials ────────────────────────────────
PROXY_URL      = os.getenv("TRIPLETEX_PROXY_URL", "https://kkpqfuj-amager.tripletex.dev/v2")
SESSION_TOKEN  = os.getenv("TRIPLETEX_SESSION_TOKEN", "YOUR_SESSION_TOKEN_HERE")
# ──────────────────────────────────────────────────────────────────────────────

TASK_PROMPT = (
    "Create a new employee with the first name 'Anna' and last name 'Hansen'. "
    "Set her email to anna.hansen@example.com."
)

def main():
    payload = {
        "prompt": TASK_PROMPT,
        "files": [],
        "tripletex_credentials": {
            "base_url": PROXY_URL,
            "session_token": SESSION_TOKEN,
        }
    }

    print(f"Sending task to /solve:\n  {TASK_PROMPT}\n")
    resp = httpx.post("http://localhost:8000/solve", json=payload, timeout=300)
    print(f"Status code: {resp.status_code}")
    print(f"Response:    {resp.json()}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "completed"
    print("\n✅ /solve returned completed successfully.")

if __name__ == "__main__":
    main()
