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
import functools
import io
import json
import re
import time

import pdfplumber
import requests as http_requests
import structlog
from google import genai
from google.genai import types
from tenacity import RetryError

from src.agent.prompt import SYSTEM_PROMPT
from src.config import settings
from src.tripletex.client import TripletexClient

log = structlog.get_logger()

MAX_ITERATIONS = 20
TIMEOUT_SECONDS = 260


# ── OpenAPI schema lookup ──────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def _load_openapi_spec() -> dict:
    """Download and cache the OpenAPI spec from Tripletex sandbox."""
    try:
        resp = http_requests.get(
            "https://kkpqfuj-amager.tripletex.dev/v2/openapi.json", timeout=30
        )
        return resp.json()
    except Exception:
        return {}


def _lookup_schema(path: str, method: str = "post") -> str:
    """Extract request body schema for a given API endpoint."""
    try:
        spec = _load_openapi_spec()
        if not spec:
            return "Schema not available"

        method = method.lower()
        endpoint = spec.get("paths", {}).get(path, {}).get(method, {})
        if not endpoint:
            # Try finding close matches
            all_paths = [p for p in spec.get("paths", {}) if path.rstrip("/") in p]
            if all_paths:
                return f"Endpoint {method.upper()} {path} not found. Similar: {', '.join(all_paths[:5])}"
            return f"No {method.upper()} endpoint found for {path}"

        schemas = spec.get("components", {}).get("schemas", {})
        results = []

        # Request body (POST/PUT)
        rb = endpoint.get("requestBody", {}).get("content", {})
        for ct, ct_def in rb.items():
            ref = ct_def.get("schema", {}).get("$ref", "")
            model_name = ref.split("/")[-1] if ref else ""
            if model_name and model_name in schemas:
                results.append(_format_model(model_name, schemas[model_name], schemas))

        # Query parameters (GET/PUT action endpoints)
        params = endpoint.get("parameters", [])
        param_lines = []
        for p in params:
            if p.get("in") == "query":
                req = " REQUIRED" if p.get("required") else ""
                ptype = p.get("schema", {}).get("type", "string")
                param_lines.append(f"  ?{p['name']}: {ptype}{req}")
        if param_lines:
            results.append("Query params:\n" + "\n".join(param_lines))

        return "\n".join(results) if results else f"No schema found for {method.upper()} {path}"
    except Exception as e:
        return f"Schema lookup error: {e}"


def _format_model(name: str, model: dict, all_schemas: dict, depth: int = 0) -> str:
    """Format a model's writable fields, resolving nested refs one level deep."""
    if depth > 1:
        return f"{name}: (nested object — use {{\"id\": <int>}})"

    required = set(model.get("required", []))
    lines = [f"Model: {name}"]
    for prop, defn in sorted(model.get("properties", {}).items()):
        if defn.get("readOnly"):
            continue
        ref = defn.get("$ref", "")
        ptype = defn.get("type", "")
        enum = defn.get("enum", [])

        if ref:
            ref_name = ref.split("/")[-1]
            ref_model = all_schemas.get(ref_name, {})
            # For reference objects, show if they just need an id
            if "id" in ref_model.get("properties", {}):
                ptype = f'{ref_name} (use {{"id": <int>}})'
            else:
                ptype = ref_name
        elif not ptype:
            ptype = "object"

        req = " REQUIRED" if prop in required else ""
        enum_str = f" enum={enum}" if enum else ""
        lines.append(f"  {prop}: {ptype}{req}{enum_str}")
    return "\n".join(lines)

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
            types.FunctionDeclaration(
                name="tripletex_schema",
                description="Look up the API schema for any Tripletex endpoint. Returns the correct field names and types. Use this when you get 422 errors about unknown fields, or before calling an unfamiliar endpoint.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "path": types.Schema(type=types.Type.STRING, description="API path e.g. '/travelExpense/cost'"),
                        "method": types.Schema(type=types.Type.STRING, description="HTTP method: 'get', 'post', or 'put'. Default: 'post'"),
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
        elif name == "tripletex_schema":
            schema_text = _lookup_schema(args["path"], args.get("method", "post"))
            return {"schema": schema_text}
        else:
            return {"error": f"Unknown tool: {name}"}
    except Exception as exc:
        # Unwrap tenacity RetryError to get the actual exception
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


import re as _re

# ── Task classifier — determines which pre-flights to run ──────────────────────

def _classify_task(prompt: str) -> set:
    """Classify task to run only needed pre-flights, saving API call budget."""
    p = prompt.lower()
    # Strip email addresses to prevent "faktura@company.no" triggering false positives
    p_clean = _re.sub(r'\S+@\S+\.\S+', '', p)

    cats = set()

    if any(w in p for w in ['salary', 'lønn', 'lön', 'lönn', 'salaire', 'nomina', 'payroll', 'wage', 'gehalt', 'payslip',
                             'salario', 'salário']):
        cats.add('payroll')

    if any(w in p for w in ['travel', 'reise', 'voyage', 'viaje', 'dienstreise', 'reisregning', 'reisrekn', 'diett', 'dietas', 'frais de d', 'nota de gastos', 'indemnit',
                             'viagem', 'nota de despesa', 'despesa de viagem']):
        cats.add('travel')

    # Pure supplier registration — classified separately to avoid invoice pre-flight waste
    if any(w in p_clean for w in ['leverand', 'lieferant', 'fournisseur', 'fornecedor', 'supplier', 'proveedor']):
        cats.add('supplier')

    # Invoice/order/credit tasks (use clean prompt; exclude expense report "abrechnung")
    _has_rechnung = 'rechnung' in p_clean and 'abrechnung' not in p
    _invoice_words = ['invoice', 'faktura', 'factura', 'facture', 'order', 'bestilling', 'commande',
                      'pedido', 'auftrag', 'credit', 'kreditnota', 'returned', 'tilbake', 'bounced',
                      'reverser', 'storniere', 'timesheet', 'timer for', 'hours for', 'horas para',
                      'stunden', 'heures pour', 'timar', 'fatura', 'inngående', 'incoming invoice']
    if any(w in p_clean for w in _invoice_words) or _has_rechnung:
        cats.add('invoice')

    # Payment registration — controls bank account + paymentType pre-flights
    if any(w in p_clean for w in ['payment', 'betaling', 'pago', 'zahlung', 'paiement', 'pagamento',
                                   'bounced', 'retur', 'avvist', 'returned', 'registrer betal']):
        cats.add('payment')

    if any(w in p for w in ['employee', 'ansatt', 'mitarbeiter', 'employ', 'empleado', 'medarbeider', 'ny ansatt', 'nouvel employ', 'nuevo empleado', 'neuen mitarbeiter', 'ansette',
                             'funcionario', 'funcionária', 'novo funcionario', 'nova funcionaria',
                             'funcionário', 'funcionária', 'novo funcionário', 'nova funcionária']):
        cats.add('employee')

    if any(w in p for w in ['voucher', 'bilag', 'journal', 'konter', 'dimensi', 'dimension',
                             'kvittering', 'bokfor', 'utlegg', 'receipt', 'bilagsforing',
                             'kontering', 'refusjon', 'reimburs', 'expense report']):
        cats.add('ledger')

    return cats


# ── Main agent loop ──────────────────────────────────────────────────────────────────────────────

async def run_agent(
    client: TripletexClient,
    prompt: str,
    files: list[dict],
    run_id: str,
) -> None:
    ai = genai.Client(api_key=settings.gemini_api_key)
    start = time.time()
    today = time.strftime("%Y-%m-%d")

    # Classify task — only run needed pre-flights to preserve API call budget
    cats = _classify_task(prompt)
    # Files attached (Tier 3) → always need accounts for voucher posting
    if files:
        cats.add('ledger')
    need_payroll = 'payroll' in cats
    need_travel = 'travel' in cats
    need_invoice = 'invoice' in cats
    need_payment = 'payment' in cats   # payment registration (subset of invoice)
    need_employee = 'employee' in cats
    need_ledger = 'ledger' in cats
    log.info("task_classified", run_id=run_id, categories=sorted(cats))

    env_hints = []

    # Division — only for payroll/employee tasks
    division_id = None
    if need_payroll or need_employee:
        try:
            div_resp = client.get("/division", params={"count": 1, "fields": "id"})
            div_values = div_resp.get("values", [])
            if div_values:
                division_id = div_values[0]["id"]
            else:
                muni_resp = client.get("/municipality", params={"number": "0301", "count": 1, "fields": "id"})
                muni_id = (muni_resp.get("values", [{}])[0].get("id") or 1)
                new_div = client.post("/division", body={
                    "name": "Hoveddivisjon", "startDate": "2020-01-01",
                    "organizationNumber": "985365785",
                    "municipality": {"id": muni_id}, "municipalityDate": "2020-01-01",
                })
                division_id = new_div.get("value", {}).get("id")
            if division_id:
                env_hints.append(f"[Division id: {division_id} — use division:{{\"id\":{division_id}}} in POST /employee/employment]")
                log.info("division_ready", run_id=run_id, division_id=division_id)
        except Exception as e:
            log.warning("division_setup_failed", run_id=run_id, error=str(e))

    # Department — only for employee/payroll tasks
    if need_employee or need_payroll:
        try:
            dept_resp = client.get("/department", params={"count": 1, "fields": "id"})
            dept_values = dept_resp.get("values", [])
            if dept_values:
                dept_id = dept_values[0]["id"]
                env_hints.append(f"[Department id: {dept_id} — POST /employee REQUIRES department:{{\"id\":{dept_id}}} on first attempt]")
                log.info("department_found", run_id=run_id, department_id=dept_id)
        except Exception as e:
            log.warning("department_discovery_failed", run_id=run_id, error=str(e))

    # Bank account + payment type — ONLY when payment registration is involved
    # (not for simple invoice creation, credit notes, supplier registration, etc.)
    if need_payment:
        try:
            acct_resp = client.get("/ledger/account", params={"number": 1920, "fields": "id,version,bankAccountNumber,isBankAccount", "count": 1})
            acct_values = acct_resp.get("values", [])
            if acct_values:
                acct = acct_values[0]
                if not acct.get("bankAccountNumber"):
                    client.put(f"/ledger/account/{acct['id']}", body={"id": acct["id"], "version": acct["version"], "bankAccountNumber": "12345678903", "isBankAccount": True})
                    log.info("bank_account_configured", run_id=run_id, account_id=acct["id"])
        except Exception as e:
            log.warning("bank_setup_failed", run_id=run_id, error=str(e))

        try:
            pt_resp = client.get("/invoice/paymentType", params={"count": 5, "fields": "id,description"})
            pt_values = pt_resp.get("values", [])
            if pt_values:
                pt_id = pt_values[0]["id"]
                env_hints.append(f"[Valid paymentTypeId: {pt_id} (use this for PUT /invoice/:payment)]")
                log.info("payment_type_found", run_id=run_id, payment_type_id=pt_id)
        except Exception as e:
            log.warning("payment_type_lookup_failed", run_id=run_id, error=str(e))

    # Travel payment type + per diem zone — only for travel tasks
    if need_travel:
        try:
            tpt_resp = client.get("/travelExpense/paymentType", params={"count": 5})
            tpt_values = tpt_resp.get("values", [])
            if tpt_values:
                tpt_id = tpt_values[0]["id"]
                env_hints.append(f"[Travel expense paymentType id: {tpt_id} (use as paymentType:{{\"id\":{tpt_id}}} in /travelExpense/cost)]")
                log.info("travel_payment_type_found", run_id=run_id, travel_payment_type_id=tpt_id)
        except Exception as e:
            log.warning("travel_payment_type_lookup_failed", run_id=run_id, error=str(e))

        try:
            zone_resp = client.get("/travelExpense/perDiemCompensationZone", params={"count": 20, "fields": "id,name"})
            zone_values = zone_resp.get("values", [])
            if zone_values:
                # Prefer domestic/innland zone; fall back to first
                domestic = next((z for z in zone_values if any(w in z.get("name", "").lower() for w in ["innland", "inland", "domestic", "stat"])), None)
                chosen_zone = domestic or zone_values[0]
                zone_id = chosen_zone["id"]
                zone_name = chosen_zone.get("name", "")
                all_zones = " | ".join(f"{z['name']}->id:{z['id']}" for z in zone_values[:5])
                env_hints.append(f"[Per diem zone (use for POST /travelExpense/perDiemCompensation): preferred={zone_name}->id:{zone_id} | all: {all_zones}]")
                log.info("per_diem_zone_found", run_id=run_id, zone_id=zone_id, zone_name=zone_name)
        except Exception as e:
            log.warning("per_diem_zone_lookup_failed", run_id=run_id, error=str(e))

    # Account IDs — only for payroll and ledger tasks (NOT for invoice/supplier/credit note)
    if need_payroll or need_ledger:
        try:
            all_accts = client.get("/ledger/account", params={"count": 1000, "fields": "id,number", "from": 0})
            acct_map = {a["number"]: a["id"] for a in all_accts.get("values", [])}
            needed = [1920, 2400, 2600, 2710, 2770, 2780, 3000, 5000,
                      6540, 6700, 6800, 6900, 7000, 7100, 7140]
            found = {n: acct_map[n] for n in needed if n in acct_map}
            if found:
                acct_hints = " | ".join(f"{n}={aid}" for n, aid in sorted(found.items()))
                env_hints.append(f"[Account IDs: {acct_hints}]")
                log.info("accounts_discovered", run_id=run_id, count=len(found))
        except Exception as e:
            log.warning("account_discovery_failed", run_id=run_id, error=str(e))

    # Salary type IDs — only for payroll, with explicit format so agent uses DB id not type#
    if need_payroll:
        try:
            st_resp = client.get("/salary/type", params={"count": 100, "fields": "id,name"})
            # Search by NAME not number — competition accounts use different type numbers
            name_map = {st["name"]: st["id"] for st in st_resp.get("values", [])}
            key_names = ["Fastlønn", "Timelønn", "Bonus", "Skattetrekk", "Timelønnet", "Sykepenger"]
            found_st = {name: name_map[name] for name in key_names if name in name_map}
            if found_st:
                st_parts = [f"{name}->id:{sid}" for name, sid in found_st.items()]
                env_hints.append(f"[SALARY TYPE DB IDs — use id: value in salaryType:{{id:X}}: {' | '.join(st_parts)}]")
                log.info("salary_types_discovered", run_id=run_id, count=len(found_st))
            else:
                # Fallback: provide all types so agent can pick
                all_types = [{"name": st["name"], "id": st["id"]} for st in st_resp.get("values", [])[:15]]
                if all_types:
                    st_parts = [f"{t['name']}->id:{t['id']}" for t in all_types]
                    env_hints.append(f"[SALARY TYPE DB IDs — use id: value in salaryType:{{id:X}}: {' | '.join(st_parts)}]")
                    log.info("salary_types_discovered", run_id=run_id, count=len(all_types))
        except Exception as e:
            log.warning("salary_type_discovery_failed", run_id=run_id, error=str(e))

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

    # Prepend today's date and environment hints so agent never has to guess
    hints = [f"[Today's date: {today}]"] + env_hints
    date_hint = types.Part.from_text(text="\n".join(hints) + "\n\n")
    user_parts = [date_hint] + file_parts + [types.Part.from_text(text=prompt)]
    contents.append(types.Content(role="user", parts=user_parts))

    nudge_count = 0
    has_made_successful_calls = False
    for iteration in range(MAX_ITERATIONS):
        if time.time() - start > TIMEOUT_SECONDS:
            log.warning("agent_timeout", run_id=run_id, iteration=iteration)
            break

        log.info("agent_iteration", run_id=run_id, iteration=iteration)

        # Call Gemini with model fallback on rate limit (429)
        # Each model has independent quota — more models = more resilience
        MODELS = [settings.gemini_model, "gemini-3-flash-preview", "gemini-2.5-flash-lite", "gemini-3.1-flash-lite-preview"]
        response = None
        retry_after_secs = 30  # default backoff

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
                    # Parse retry-after hint from error
                    match = re.search(r'retry in (\d+(?:\.\d+)?)s', err)
                    if match:
                        retry_after_secs = max(retry_after_secs, int(float(match.group(1))) + 5)
                    continue  # try next model
                else:
                    raise

        if response is None:
            # All models rate limited — wait using the parsed retry-after, then try once more
            wait_time = min(retry_after_secs, 90)  # cap at 90s to stay within 5-min timeout
            log.warning("all_models_rate_limited", run_id=run_id, wait=wait_time)
            await asyncio.sleep(wait_time)
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
                    log.info("using_model_after_wait", run_id=run_id, model=model)
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

        # No function calls → done (but nudge once if agent hasn't done any work yet)
        if not function_calls:
            if not has_made_successful_calls and nudge_count < 1 and prompt.strip().lower() != "test":
                nudge_count += 1
                log.warning("no_tools_nudge", run_id=run_id, iteration=iteration, nudge=nudge_count)
                contents.append(types.Content(role="user", parts=[
                    types.Part.from_text(text="You did not call any tools. Please use the tripletex_post/tripletex_get tools to complete the task. Do not just describe what to do — actually call the API.")
                ]))
                continue
            log.info("agent_complete", run_id=run_id, iterations=iteration + 1)
            break

        # Execute tools and collect responses
        tool_response_parts = []
        for fc in function_calls:
            args = dict(fc.args)
            log.info("tool_call", run_id=run_id, tool=fc.name, inputs=args)
            result = _execute_tool(fc.name, args, client)
            log.info("tool_result", run_id=run_id, tool=fc.name,
                     result_preview=json.dumps(result, default=str)[:500])

            # Track if any tool call succeeded (no error)
            if "error" not in result:
                has_made_successful_calls = True

            tool_response_parts.append(
                types.Part.from_function_response(
                    name=fc.name,
                    response={"result": json.dumps(result, default=str)},
                )
            )

        contents.append(types.Content(role="user", parts=tool_response_parts))
