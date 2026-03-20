import asyncio
import os
from src.tripletex.client import TripletexClient
from src.agent.runner import run_agent

async def main():
    client = TripletexClient(
        os.environ.get("TRIPLETEX_PROXY_URL", ""),
        os.environ.get("TRIPLETEX_SESSION_TOKEN", "")
    )
    await run_agent(
        client=client,
        prompt='Crie o projeto "Migracao Montanha" vinculado ao cliente Montanha Lda (org. nr 986713344). O gerente de projeto e Bruno Pereira (bruno.pereira@example.org).',
        files=[],
        run_id="debug-project"
    )

asyncio.run(main())
