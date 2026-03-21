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

def _fix_date(d: str) -> str:
    """Fix invalid dates like Feb 29 in non-leap years."""
    import calendar
    if not isinstance(d, str) or len(d) != 10:
        return d
    try:
        parts = d.split("-")
        y, m, day = int(parts[0]), int(parts[1]), int(parts[2])
        max_day = calendar.monthrange(y, m)[1]
        if day > max_day:
            return f"{y:04d}-{m:02d}-{max_day:02d}"
    except (ValueError, IndexError):
        pass
    return d


def _fix_fields_param(path: str, params: dict | None) -> dict | None:
    """Fix invalid field names for specific DTOs."""
    if not params or "fields" not in params:
        return params
    fields = params["fields"]
    if "occupationCode" in path and "name" in fields.split(","):
        params["fields"] = fields.replace("name", "nameNO")
    elif "/currency" in path and "name" in fields.split(","):
        params["fields"] = fields.replace("name", "code")
    return params


def _execute_tool(name: str, args: dict, client: TripletexClient) -> dict:
    try:
        if name == "tripletex_get":
            params = _fix_fields_param(args["path"], args.get("params"))
            return client.get(args["path"], params=params)
        elif name == "tripletex_post":
            body = args["body"]
            path = args.get("path", "")
            # Auto-fix: strip invalid fields from /activity POST, default activityType
            if path == "/activity":
                body.pop("isGeneralActivity", None)
                body.pop("isProjectActivity", None)
                body.setdefault("activityType", "PROJECT_GENERAL_ACTIVITY")
            # Auto-fix: dates (e.g. Feb 29 in non-leap year)
            if "date" in body:
                body["date"] = _fix_date(body["date"])
            # Auto-fix: ensure amountGross == amountGrossCurrency on every voucher posting
            if "postings" in body and isinstance(body["postings"], list):
                for posting in body["postings"]:
                    if isinstance(posting, dict) and "amountGrossCurrency" in posting:
                        posting.setdefault("amountGross", posting["amountGrossCurrency"])
                        posting.setdefault("currency", {"id": 1})
                # Auto-fix: ensure postings sum to 0 (fix rounding on last posting)
                total = sum(p.get("amountGrossCurrency", 0) for p in body["postings"] if isinstance(p, dict))
                if total != 0 and len(body["postings"]) >= 2:
                    last = body["postings"][-1]
                    last["amountGrossCurrency"] = round(last.get("amountGrossCurrency", 0) - total, 2)
                    last["amountGross"] = last["amountGrossCurrency"]
            # Auto-fix: for employee employment, ensure dateOfBirth is set first
            if path == "/employee/employment" and "employee" in body:
                emp_id = body["employee"].get("id")
                if emp_id:
                    try:
                        emp_resp = client.get("/employee/" + str(emp_id), params={"fields": "id,version,dateOfBirth"})
                        emp_data = emp_resp.get("value", emp_resp)
                        if not emp_data.get("dateOfBirth"):
                            client.put(f"/employee/{emp_id}", body={
                                "id": emp_id, "version": emp_data.get("version", 1),
                                "dateOfBirth": "1990-01-15"
                            })
                            log.info("auto_set_dateOfBirth", employee_id=emp_id)
                    except Exception:
                        pass  # best-effort, don't block
            return client.post(args["path"], body=body, params=args.get("params"))
        elif name == "tripletex_put":
            params = args.get("params") or {}
            # Auto-fix: ensure paidAmountCurrency = paidAmount on payment calls
            if "/:payment" in args.get("path", "") and "paidAmount" in params:
                params.setdefault("paidAmountCurrency", params["paidAmount"])
            body = args.get("body") or {}
            if "date" in body:
                body["date"] = _fix_date(body["date"])
            return client.put(args["path"], body=body, params=params)
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

    if any(w in p for w in ['salary', 'lønn', 'løn', 'lön', 'lönn', 'salaire', 'nomina', 'payroll', 'wage', 'gehalt', 'payslip',
                             'salario', 'salário', 'grunnløn', 'grunnlön', 'eingongsbonus', 'bonus',
                             'køyr løn', 'kjør lønn', 'kör lön']):
        cats.add('payroll')

    if any(w in p for w in ['travel', 'reise', 'voyage', 'viaje', 'dienstreise', 'reisregning', 'reisrekn', 'diett', 'dietas', 'frais de d', 'nota de gastos', 'indemnit',
                             'viagem', 'nota de despesa', 'despesa de viagem']):
        cats.add('travel')

    # Pure supplier registration — classified separately to avoid invoice pre-flight waste
    if any(w in p_clean for w in ['leverand', 'lieferant', 'fournisseur', 'fornecedor', 'supplier', 'proveedor']):
        cats.add('supplier')
        # Supplier invoices always need ledger accounts for voucher posting
        if any(w in p_clean for w in ['invoice', 'faktura', 'factura', 'vat', 'mva', 'rechnung',
                                       'register', 'book', 'inngående', 'incoming']):
            cats.add('ledger')

    # Bank reconciliation — needs payment+ledger+invoice+supplier pre-flights
    if any(w in p for w in ['reconcil', 'rapproch', 'avstem', 'bankrekonsil', 'kontoutskrift',
                              'bankutskrift', 'bank statement', 'releve bancaire', 'extracto bancario',
                              'extrato bancario', 'concilia']):
        cats.add('payment')
        cats.add('ledger')
        cats.add('invoice')
        cats.add('supplier')

    # Ledger analysis → project creation (libro mayor / Hauptbuch analysis to identify top accounts)
    if any(w in p for w in ['libro mayor', 'hauptbuch', 'grand livre', 'general ledger', 'livro razão',
                              'analice el', 'analise o', 'analyz', 'analysier',
                              'størst økning', 'biggest increase', 'mayor increment',
                              'cuentas de gastos', 'kostnadskon', 'incremento en monto',
                              'maior aumento', 'hovudbok', 'størst auke',
                              'kostnadskonto', 'ausgabenkont']):
        cats.add('ledger')
        cats.add('employee')  # need project manager ID

    # Ledger error correction
    if any(w in p for w in ['fehler', 'feil i', 'korrektur', 'korriger', 'correct', 'correction',
                              'doppel', 'duplik', 'falsche', 'manglende mva', 'feil konto',
                              'wrong account', 'duplicate', 'missing vat']):
        cats.add('ledger')

    # Month-end closing
    if any(w in p for w in ['månedsavslut', 'månavslutn', 'monatsabschluss', 'month-end',
                              'encerramento mensal', 'lønnsavsetj', 'lønnsavsetn']):
        cats.add('ledger')

    # Project lifecycle (full workflow with budget + hours + supplier + invoice)
    if any(w in p for w in ['ciclo de vida', 'project lifecycle', 'presupuesto', 'budget']):
        cats.add('invoice')
        cats.add('payment')
        cats.add('ledger')
        cats.add('employee')

    # Reminder fee / overdue invoice
    if any(w in p for w in ['overdue', 'purring', 'purregebyr', 'reminder fee', 'mahnung', 'mahngebuh', 'mahngebüh',
                              'forfalt', 'forfallen', 'forfalne', 'uteståande', 'utestående',
                              'pago atrasado', 'vencido', 'vencida', 'en retard',
                              'überfällig', 'uberfallig', 'delbetaling', 'partial payment']):
        cats.add('invoice')
        cats.add('payment')
        cats.add('ledger')

    # Invoice/order/credit tasks (use clean prompt; exclude expense report "abrechnung")
    _has_rechnung = 'rechnung' in p_clean and 'abrechnung' not in p
    _invoice_words = ['invoice', 'faktura', 'factura', 'facture', 'order', 'bestilling', 'commande',
                      'pedido', 'auftrag', 'credit', 'kreditnota', 'returned', 'tilbake', 'bounced',
                      'reverser', 'storniere', 'timesheet', 'timer for', 'hours for', 'horas para',
                      'stunden', 'heures pour', 'timar', 'fatura', 'inngående', 'incoming invoice',
                      'reconcil', 'rapproch', 'avstem', 'bankutskrift', 'relev']
    if any(w in p_clean for w in _invoice_words) or _has_rechnung:
        cats.add('invoice')

    # Payment registration — controls bank account + paymentType pre-flights
    if any(w in p_clean for w in ['payment', 'betaling', 'pago', 'zahlung', 'paiement', 'pagamento',
                                   'bounced', 'retur', 'avvist', 'returned', 'registrer betal',
                                   'agio', 'valuta', 'exchange rate', 'tipo de cambio', 'tipo de câmbio',
                                   'currency', 'kurs', 'wechselkurs', 'disagio',
                                   'reconcil', 'rapproch', 'avstem', 'bankutskrift', 'relev']):
        cats.add('payment')

    if any(w in p for w in ['employee', 'ansatt', 'mitarbeiter', 'employ', 'empleado', 'medarbeider', 'ny ansatt', 'nouvel employ', 'nuevo empleado', 'neuen mitarbeiter', 'ansette',
                             'funcionario', 'funcionária', 'novo funcionario', 'nova funcionaria',
                             'funcionário', 'funcionária', 'novo funcionário', 'nova funcionária',
                             'tilsett', 'tilsette', 'ny tilsett', 'medarbeidar', 'onboarding',
                             'tilbodsbrev', 'tilbudsbrev', 'arbeidskontrakt', 'contrato de trabalho',
                             'contrato de empleo', 'incorporacion', 'angestellt']):
        cats.add('employee')

    if any(w in p for w in ['voucher', 'bilag', 'journal', 'konter', 'dimensi', 'dimension',
                             'kvittering', 'bokfor', 'bokfør', 'utlegg', 'receipt', 'bilagsforing',
                             'kontering', 'refusjon', 'reimburs', 'expense report',
                             'avskriv', 'årsoppgjer', 'årsoppgjør', 'depreciation',
                             'skattekostnad', 'periodisering', 'abschluss', 'year-end',
                             'forskotsbetalt', 'prepaid', 'accrual', 'encerramento anual',
                             'agio', 'valutagevinst', 'valutatap', 'exchange rate', 'tipo de cambio',
                             'månedsavslut', 'månavslutn', 'lønnsavsetj', 'lønnsavsetn',
                             'korrektur', 'korriger', 'correction', 'feil i bilag',
                             'quittung', 'reçu', 'recibo', 'ausgabe', 'beleg',
                             'purregebyr', 'purring', 'reminder fee', 'mahnung',
                             'forfalt', 'forfallen', 'overdue', 'uteståande', 'utestående',
                             # French
                             'amortissement', 'cloture', 'clôture', 'exercice', 'comptabilis',
                             'impôt', 'impot', 'charge constat', 'ecriture', 'écriture',
                             'rapprochement', 'grand livre', 'mensuel',
                             # Portuguese
                             'depreciação', 'depreciaçao', 'encerramento', 'imposto',
                             'provisão', 'provisao', 'lançamento', 'lancamento',
                             # Spanish
                             'depreciación', 'depreciacion', 'cierre', 'asiento',
                             'provisión', 'provision', 'amortización', 'amortizacion',
                             # German
                             'abschreibung', 'jahresabschluss', 'monatsabschluss', 'buchung',
                             'steuer', 'rückstellung', 'ruckstellung']):
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

    # Department — for employee/payroll/receipt/ledger tasks (receipts often specify department)
    if need_employee or need_payroll or need_ledger:
        try:
            dept_resp = client.get("/department", params={"count": 50, "fields": "id,name"})
            dept_values = dept_resp.get("values", [])
            if dept_values:
                dept_id = dept_values[0]["id"]
                env_hints.append(f"[Department id: {dept_id} — POST /employee REQUIRES department:{{\"id\":{dept_id}}} on first attempt]")
                dept_list = ", ".join(f"{d.get('name','?')}={d['id']}" for d in dept_values)
                env_hints.append(f"[All departments: {dept_list}]")
                log.info("department_found", run_id=run_id, department_count=len(dept_values))
        except Exception as e:
            log.warning("department_discovery_failed", run_id=run_id, error=str(e))

    # Bank account + payment type — needed for payment AND invoice creation
    if need_payment or need_invoice:
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
            zone_resp = client.get("/travelExpense/perDiemCompensationZone", params={"count": 20})
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

        # Pre-fetch cost categories so agent doesn't need to GET them
        try:
            cc_resp = client.get("/travelExpense/costCategory", params={"showOnTravelExpenses": True, "count": 50, "fields": "id,description"})
            cc_values = cc_resp.get("values", [])
            if cc_values:
                cc_list = ", ".join(f"{c.get('description','?')}={c['id']}" for c in cc_values)
                env_hints.append(f"[Travel cost categories: {cc_list}]")
                log.info("travel_cost_categories_found", run_id=run_id, count=len(cc_values))
        except Exception as e:
            log.warning("travel_cost_categories_failed", run_id=run_id, error=str(e))

    # Account IDs — only for payroll and ledger tasks (NOT for invoice/supplier/credit note)
    if need_payroll or need_ledger:
        try:
            all_accts = client.get("/ledger/account", params={"count": 1000, "fields": "id,number", "from": 0})
            acct_map = {a["number"]: a["id"] for a in all_accts.get("values", [])}
            needed = [1500, 1700, 1710, 1920, 2400, 2600, 2710, 2770, 2780, 2900, 2920,
                      3000, 3400, 4300, 5000, 6010, 6540, 6700, 6800, 6900, 7000, 7100, 7140, 8060, 8071]

            # Also pull any 4-digit account numbers the task explicitly mentions
            task_acct_nums = set(int(m) for m in _re.findall(r'\b([1-9]\d{3})\b', prompt)
                                 if 1000 <= int(m) <= 9999)
            all_needed = sorted(set(needed) | task_acct_nums)

            found = {n: acct_map[n] for n in all_needed if n in acct_map}
            if found:
                acct_hints = " | ".join(f"{n}={aid}" for n, aid in sorted(found.items()))
                env_hints.append(f"[Account IDs: {acct_hints}]")

            # For task accounts not found, report nearest alternative so agent doesn't waste GETs
            missing_hints = []
            for num in sorted(task_acct_nums):
                if num not in acct_map:
                    # nearest in same 100-range, then same 1000-range
                    candidates = [(n, aid) for n, aid in acct_map.items() if n // 100 == num // 100]
                    if not candidates:
                        candidates = [(n, aid) for n, aid in acct_map.items() if n // 1000 == num // 1000]
                    if candidates:
                        nearest = min(candidates, key=lambda x: abs(x[0] - num))
                        missing_hints.append(f"{num}=NOT FOUND→use {nearest[0]}({nearest[1]})")
                    else:
                        missing_hints.append(f"{num}=NOT FOUND")
            if missing_hints:
                env_hints.append(f"[Missing task accounts — use nearest: {' | '.join(missing_hints)}]")

            log.info("accounts_discovered", run_id=run_id, count=len(found), missing=len(missing_hints))
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

        # Auto-detect MIME type from filename extension when not provided
        if not mime:
            fl = fname.lower()
            if fl.endswith(".pdf"):
                mime = "application/pdf"
            elif fl.endswith(".csv"):
                mime = "text/csv"
            elif fl.endswith(".png"):
                mime = "image/png"
            elif fl.endswith((".jpg", ".jpeg")):
                mime = "image/jpeg"

        if b64_data:
            raw_bytes = base64.b64decode(b64_data)

            # Fallback: detect by magic bytes / content when extension gives no clue
            if not mime:
                if raw_bytes[:4] == b"%PDF":
                    mime = "application/pdf"
                else:
                    try:
                        sample = raw_bytes[:512].decode("utf-8")
                        if ";" in sample or "," in sample:
                            mime = "text/csv"
                    except UnicodeDecodeError:
                        pass

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
    consecutive_errors = 0
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

        async def _call_gemini(model: str):
            """Run synchronous Gemini call in thread pool with 120s timeout."""
            cfg = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, tools=TOOLS)
            fn = functools.partial(ai.models.generate_content, model=model, contents=contents, config=cfg)
            return await asyncio.wait_for(asyncio.to_thread(fn), timeout=120)

        for model in MODELS:
            try:
                response = await _call_gemini(model)
                if iteration == 0:
                    log.info("using_model", run_id=run_id, model=model)
                break
            except asyncio.TimeoutError:
                log.warning("gemini_timeout", run_id=run_id, model=model)
                continue  # try next model
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
                    response = await _call_gemini(model)
                    log.info("using_model_after_wait", run_id=run_id, model=model)
                    break
                except (asyncio.TimeoutError, Exception):
                    continue
            if response is None:
                log.error("gemini_failed_all_models", run_id=run_id)
                break

        # Guard against empty/blocked responses
        candidate_content = response.candidates[0].content if response.candidates else None
        candidate_parts = getattr(candidate_content, "parts", None) if candidate_content else None
        if not response.candidates or candidate_content is None or not candidate_parts:
            finish_reason = str(response.candidates[0].finish_reason) if response.candidates else "no_candidates"
            log.warning("gemini_empty_response", run_id=run_id, iteration=iteration, finish_reason=finish_reason)
            # MALFORMED_FUNCTION_CALL → nudge and retry instead of stopping
            if "MALFORMED" in finish_reason and nudge_count < 3:
                nudge_count += 1
                contents.append(types.Content(role="user", parts=[
                    types.Part.from_text(
                        text="Your previous function call was malformed and could not be parsed. "
                             "Please call ONE tool at a time with simple, valid arguments. "
                             "Continue with the next pending action from the task.")
                ]))
                continue
            break

        # Add model response to history
        contents.append(candidate_content)

        # Collect function calls
        function_calls = [
            part.function_call
            for part in candidate_parts
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
                consecutive_errors = 0
            else:
                consecutive_errors += 1

            # Break early if agent is stuck in an error loop (same error 4+ times)
            if consecutive_errors >= 4:
                log.warning("error_loop_break", run_id=run_id, consecutive_errors=consecutive_errors)
                # Inject a nudge to try a different approach
                tool_response_parts.append(
                    types.Part.from_function_response(
                        name=fc.name,
                        response={"result": json.dumps({
                            "error": "STUCK IN LOOP — you have made 4+ consecutive failing calls. "
                                     "Try a COMPLETELY DIFFERENT approach: use tripletex_schema to discover "
                                     "correct fields, or skip this step and move to the next part of the task."
                        })},
                    )
                )
                consecutive_errors = 0
                break

            tool_response_parts.append(
                types.Part.from_function_response(
                    name=fc.name,
                    response={"result": json.dumps(result, default=str)},
                )
            )

        contents.append(types.Content(role="user", parts=tool_response_parts))
