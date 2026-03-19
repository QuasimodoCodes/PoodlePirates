"""
Agent runner — Claude tool-calling loop.

Flow:
  1. Build messages with the task prompt (+ any file content).
  2. Send to Claude with Tripletex tools.
  3. Claude returns tool_use blocks → execute each against TripletexClient.
  4. Feed tool results back to Claude.
  5. Repeat until Claude returns stop_reason == "end_turn" (task done).
"""
import json
import time

import anthropic
import structlog

from src.agent.prompt import SYSTEM_PROMPT
from src.agent.tools import TOOLS
from src.config import settings
from src.tripletex.client import TripletexClient

log = structlog.get_logger()

MAX_ITERATIONS = 20        # hard cap on tool-call rounds
TIMEOUT_SECONDS = 260      # stop looping after ~4.3 min (platform allows 5 min)


def _execute_tool(name: str, inputs: dict, client: TripletexClient) -> str:
    """Execute a single tool call and return the result as a JSON string."""
    try:
        if name == "tripletex_get":
            result = client.get(inputs["path"], params=inputs.get("params"))
        elif name == "tripletex_post":
            result = client.post(inputs["path"], body=inputs["body"])
        elif name == "tripletex_put":
            result = client.put(inputs["path"], body=inputs["body"], params=inputs.get("params"))
        elif name == "tripletex_delete":
            client.delete(inputs["path"])
            result = {"status": "deleted"}
        else:
            result = {"error": f"Unknown tool: {name}"}
    except Exception as exc:
        result = {"error": str(exc)}

    return json.dumps(result, ensure_ascii=False, default=str)


def _build_initial_messages(prompt: str, files: list[dict]) -> list[dict]:
    """Build the initial user message, optionally including file content."""
    content: list = []

    # Attach any files as base64 images or extracted text
    for f in files:
        mime = f.get("mime_type", "")
        b64 = f.get("content_base64", "")
        name = f.get("name", "file")

        if mime.startswith("image/") and b64:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": mime, "data": b64},
            })
            content.append({"type": "text", "text": f"(File: {name})"})
        elif b64:
            # Non-image: treat as text context if we have extracted text
            text = f.get("text", "")
            if text:
                content.append({"type": "text", "text": f"File '{name}':\n{text}"})

    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


async def run_agent(
    client: TripletexClient,
    prompt: str,
    files: list[dict],
    run_id: str,
) -> None:
    """
    Run the Claude tool-calling loop until the task is complete.
    """
    ai = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    messages = _build_initial_messages(prompt, files)
    start = time.time()

    for iteration in range(MAX_ITERATIONS):
        if time.time() - start > TIMEOUT_SECONDS:
            log.warning("agent_timeout", run_id=run_id, iteration=iteration)
            break

        log.info("agent_iteration", run_id=run_id, iteration=iteration, messages=len(messages))

        response = ai.messages.create(
            model=settings.anthropic_model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        log.info(
            "claude_response",
            run_id=run_id,
            stop_reason=response.stop_reason,
            tool_calls=[b.name for b in response.content if b.type == "tool_use"],
        )

        # Add Claude's response to the conversation
        messages.append({"role": "assistant", "content": response.content})

        # Task complete — no more tool calls
        if response.stop_reason == "end_turn":
            log.info("agent_complete", run_id=run_id, iterations=iteration + 1)
            break

        # Execute all tool calls Claude requested
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                log.info("tool_call", run_id=run_id, tool=block.name, inputs=block.input)
                result_str = _execute_tool(block.name, block.input, client)
                log.info("tool_result", run_id=run_id, tool=block.name, result_preview=result_str[:200])

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_str,
                })

            messages.append({"role": "user", "content": tool_results})
        else:
            # Unexpected stop reason
            log.warning("unexpected_stop_reason", run_id=run_id, reason=response.stop_reason)
            break
