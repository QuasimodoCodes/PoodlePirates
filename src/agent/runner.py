"""
Agent runner — Gemini tool-calling loop (google-genai SDK).

Flow:
  1. Build the initial message with the task prompt (+ any file content).
  2. Send to Gemini with Tripletex tools defined as function declarations.
  3. Gemini returns function_call parts → execute each against TripletexClient.
  4. Feed tool results back to Gemini.
  5. Repeat until Gemini returns a plain text response (task done).
"""
import asyncio
import base64
import io
import json
import time

import structlog
from google import genai
from google.genai import types

from src.agent.prompt import SYSTEM_PROMPT
from src.config import settings
from src.tripletex.client import TripletexClient

log = structlog.get_logger()

MAX_ITERATIONS = 20
TIMEOUT_SECONDS = 260

# ── Gemini tool declarations ───────────────────────────────────────────────────

TOOLS = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="tripletex_get",
                description="Call a Tripletex API GET endpoint to read or search for data. Use to look up IDs before creating resources.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "path": types.Schema(type=types.Type.STRING, description="API path e.g. '/employee'. Do NOT include the base URL."),
                        "params": types.Schema(type=types.Type.OBJECT, description="Optional query parameters e.g. {\"name\": \"Acme\", \"count\": 5}."),
                    },
                    required=["path"],
                ),
            ),
            types.FunctionDeclaration(
                name="tripletex_post",
                description="Call a Tripletex API POST endpoint to create a new resource. Returns the created object with its new id.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "path": types.Schema(type=types.Type.STRING, description="API path e.g. '/employee'."),
                        "body": types.Schema(type=types.Type.OBJECT, description="JSON body of the resource to create."),
                        "params": types.Schema(type=types.Type.OBJECT, description="Optional query parameters e.g. {\"sendToCustomer\": true}."),
                    },
                    required=["path", "body"],
                ),
            ),
            types.FunctionDeclaration(
                name="tripletex_put",
                description="Call a Tripletex API PUT endpoint to update an existing resource. Include the id in the path.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "path": types.Schema(type=types.Type.STRING, description="API path with id e.g. '/employee/42'."),
                        "body": types.Schema(type=types.Type.OBJECT, description="Full updated resource body."),
                        "params": types.Schema(type=types.Type.OBJECT, description="Optional query parameters."),
                    },
                    required=["path", "body"],
                ),
            ),
            types.FunctionDeclaration(
                name="tripletex_delete",
                description="Call a Tripletex API DELETE endpoint to remove a resource.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "path": types.Schema(type=types.Type.STRING, description="API path with id e.g. '/travelExpense/99'."),
                    },
                    required=["path"],
                ),
            ),
        ]
    )
]


# ── Tool execution ─────────────────────────────────────────────────────────────

def _execute_tool(name: str, args: dict, client: TripletexClient) -> dict:
    try:
        if name == "tripletex_get":
            return client.get(args["path"], params=args.get("params"))
        elif name == "tripletex_post":
            return client.post(args["path"], body=args["body"], params=args.get("params"))
        elif name == "tripletex_put":
            return client.put(args["path"], body=args["body"], params=args.get("params"))
        elif name == "tripletex_delete":
            client.delete(args["path"])
            return {"status": "deleted"}
        else:
            return {"error": f"Unknown tool: {name}"}
    except Exception as exc:
        # Unwrap tenacity RetryError to get the actual exception
        from tenacity import RetryError
        actual = exc
        if isinstance(exc, RetryError) and exc.last_attempt.failed:
            actual = exc.last_attempt.exception()

        detail = str(actual)
        if hasattr(actual, "response"):
            try:
                detail = actual.response.json()
            except Exception:
                try:
                    detail = actual.response.text
                except Exception:
                    pass
        return {"error": detail}


# ── Main agent loop ────────────────────────────────────────────────────────────

async def run_agent(
    client: TripletexClient,
    prompt: str,
    files: list[dict],
    run_id: str,
) -> None:
    ai = genai.Client(api_key=settings.gemini_api_key)
    start = time.time()
    today = time.strftime("%Y-%m-%d")

    # Build initial contents
    contents: list[types.Content] = []

    # Attach files if present — PDFs are extracted to text, images sent inline
    file_parts = []
    for f in files:
        mime = f.get("mime_type", "")
        b64_data = f.get("content_base64", "")
        text_content = f.get("text", "")
        fname = f.get("filename", f.get("name", "file"))

        if b64_data:
            raw_bytes = base64.b64decode(b64_data)

            if mime == "application/pdf":
                # Extract text from PDF pages
                import pdfplumber
                extracted = []
                try:
                    with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                extracted.append(page_text)
                except Exception as e:
                    log.warning("pdf_extraction_failed", file=fname, error=str(e))
                pdf_text = "\n".join(extracted)
                if pdf_text.strip():
                    file_parts.append(types.Part.from_text(
                        text=f"File '{fname}' (PDF content):\n{pdf_text}"))
                else:
                    # Scanned PDF with no text — send as image to Gemini
                    file_parts.append(types.Part.from_bytes(data=raw_bytes, mime_type=mime))
                    file_parts.append(types.Part.from_text(
                        text=f"(File: {fname} — scanned PDF, extract data from the image)"))
            elif mime.startswith("image/"):
                file_parts.append(types.Part.from_bytes(data=raw_bytes, mime_type=mime))
                file_parts.append(types.Part.from_text(text=f"(File: {fname})"))
            else:
                # Try decoding as text (CSV, JSON, etc.)
                try:
                    decoded_text = raw_bytes.decode("utf-8")
                    file_parts.append(types.Part.from_text(
                        text=f"File '{fname}':\n{decoded_text}"))
                except UnicodeDecodeError:
                    file_parts.append(types.Part.from_bytes(
                        data=raw_bytes, mime_type=mime or "application/octet-stream"))
                    file_parts.append(types.Part.from_text(text=f"(File: {fname})"))
        elif text_content:
            file_parts.append(types.Part.from_text(text=f"File '{fname}':\n{text_content}"))

    # Prepend today's date so agent never has to guess
    date_hint = types.Part.from_text(text=f"[Today's date: {today}]\n\n")
    user_parts = [date_hint] + file_parts + [types.Part.from_text(text=prompt)]
    contents.append(types.Content(role="user", parts=user_parts))

    for iteration in range(MAX_ITERATIONS):
        if time.time() - start > TIMEOUT_SECONDS:
            log.warning("agent_timeout", run_id=run_id, iteration=iteration)
            break

        log.info("agent_iteration", run_id=run_id, iteration=iteration)

        # Call Gemini with model fallback on rate limit (429)
        # gemini-3-flash-preview has better instruction following than flash-lite
        MODELS = [settings.gemini_model, "gemini-3-flash-preview", "gemini-2.5-flash-lite"]
        response = None
        for model in MODELS:
            try:
                response = ai.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        tools=TOOLS,
                    ),
                )
                if iteration == 0:
                    log.info("using_model", run_id=run_id, model=model)
                break
            except Exception as e:
                err = str(e)
                if "429" in err or "quota" in err.lower() or "rate" in err.lower():
                    log.warning("gemini_rate_limit", run_id=run_id, model=model)
                    continue  # try next model
                else:
                    raise

        if response is None:
            # All models rate limited — wait and retry with each model again
            log.warning("all_models_rate_limited", run_id=run_id, wait=15)
            await asyncio.sleep(15)
            for model in MODELS:
                try:
                    response = ai.models.generate_content(
                        model=model,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            tools=TOOLS,
                        ),
                    )
                    break
                except Exception:
                    continue
            if response is None:
                log.error("gemini_failed_all_models", run_id=run_id)
                break

        # Guard against empty/blocked responses
        if not response.candidates or response.candidates[0].content is None:
            log.warning("gemini_empty_response", run_id=run_id, iteration=iteration,
                        finish_reason=response.candidates[0].finish_reason if response.candidates else "no_candidates")
            break

        # Add model response to history
        contents.append(response.candidates[0].content)

        # Collect function calls
        function_calls = [
            part.function_call
            for part in response.candidates[0].content.parts
            if part.function_call is not None
        ]

        log.info("gemini_response", run_id=run_id, function_calls=[fc.name for fc in function_calls])

        # No function calls → done
        if not function_calls:
            log.info("agent_complete", run_id=run_id, iterations=iteration + 1)
            break

        # Execute tools and collect responses
        tool_response_parts = []
        for fc in function_calls:
            args = dict(fc.args)
            log.info("tool_call", run_id=run_id, tool=fc.name, inputs=args)
            result = _execute_tool(fc.name, args, client)
            log.info("tool_result", run_id=run_id, tool=fc.name,
                     result_preview=json.dumps(result, default=str)[:200])

            tool_response_parts.append(
                types.Part.from_function_response(
                    name=fc.name,
                    response={"result": json.dumps(result, default=str)},
                )
            )

        contents.append(types.Content(role="user", parts=tool_response_parts))
