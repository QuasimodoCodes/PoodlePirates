import asyncio
from src.tripletex.client import TripletexClient
from src.agent.runner import run_agent

async def main():
    client = TripletexClient(
        "https://tx-proxy-jwanbnu3pq-lz.a.run.app/v2",
        "JbFWuZmXV6W48IVrEYKWCb9TsXNqKrDlJUeg97Hi_qk"
    )
    await run_agent(
        client=client,
        prompt='Crie o projeto "Migracao Montanha" vinculado ao cliente Montanha Lda (org. nr 986713344). O gerente de projeto e Bruno Pereira (bruno.pereira@example.org).',
        files=[],
        run_id="debug-project"
    )

asyncio.run(main())
