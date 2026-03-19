"""
Agent runner — placeholder for Step 3.
Will be replaced with the full Claude tool-calling loop.
"""
from src.tripletex.client import TripletexClient


async def run_agent(client: TripletexClient, prompt: str, files: list[dict], run_id: str) -> None:
    """
    Interpret `prompt` and execute the required Tripletex API calls.
    Placeholder: just fetches /company to prove connectivity.
    Full implementation arrives in Step 3.
    """
    company = client.get("/company")
    name = company.get("value", {}).get("name", "unknown")
    print(f"[{run_id}] Connected to company: {name}")
    print(f"[{run_id}] Task: {prompt[:200]}")
    # TODO: Step 3 — Claude tool-calling loop goes here
