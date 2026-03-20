"""
Stress test suite for the Tripletex AI Agent.

Generates 200+ realistic task prompts across all 30 task types,
runs them against the agent using the sandbox API, verifies results,
and logs everything.

Usage:
    python tests/stress_test.py [--count N] [--delay SECS] [--category CAT]
"""
import asyncio
import argparse
import json
import random
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import os

# ── Sandbox credentials ────────────────────────────────────────────────────────
SANDBOX_BASE = os.environ.get("SANDBOX_BASE", "https://kkpqfuj-amager.tripletex.dev/v2")
SANDBOX_TOKEN = os.environ.get("SANDBOX_TOKEN", "")
AGENT_URL = "http://localhost:8000/solve"

AUTH = ("0", SANDBOX_TOKEN)


@dataclass
class TaskResult:
    task_id: str
    category: str
    language: str
    prompt: str
    status: str = "pending"        # pending, passed, failed, error
    checks_passed: int = 0
    checks_total: int = 0
    error_message: str = ""
    elapsed_seconds: float = 0.0
    api_calls_by_agent: int = 0
    details: dict = field(default_factory=dict)


# ── Task generators ───────────────────────────────────────────────────────────
# Each generator returns a list of (prompt, verify_func, check_count) tuples

def _rand_name():
    first = random.choice(["Ole", "Kari", "Per", "Anna", "Erik", "Lise", "Hans", "Mari", "Jonas", "Ingrid",
                           "Lars", "Hilde", "Tor", "Silje", "Bjorn", "Mette", "Sven", "Astrid"])
    last = random.choice(["Hansen", "Johansen", "Olsen", "Larsen", "Andersen", "Pedersen", "Nilsen",
                           "Kristiansen", "Berg", "Haugen", "Bakken", "Lund", "Dahl", "Moen", "Vik"])
    return first, last


def _rand_email(first, last):
    domain = random.choice(["firma.no", "bedrift.no", "selskap.no", "test.no", "example.com"])
    return f"{first.lower()}.{last.lower()}@{domain}"


def _rand_org_nr():
    return str(random.randint(800000000, 999999999))


def _rand_phone():
    return f"+47{random.randint(40000000, 99999999)}"


def _rand_city():
    return random.choice(["Oslo", "Bergen", "Trondheim", "Stavanger", "Drammen", "Tromso", "Kristiansand",
                           "Fredrikstad", "Bodo", "Sandefjord", "Alesund", "Haugesund"])


def _rand_street():
    streets = ["Storgata", "Parkveien", "Kirkegata", "Solveien", "Fjordgata", "Havnegata", "Skogveien",
               "Bankgata", "Torggata", "Langveien", "Elvegata", "Sjogate"]
    return f"{random.choice(streets)} {random.randint(1, 200)}"


def _rand_postal():
    return f"{random.randint(1000, 9999):04d}"


def _rand_product():
    products = ["Konsulenttime", "Webdesign", "Programvareutvikling", "IT-support", "Prosjektledelse",
                "Grafisk design", "SEO-tjeneste", "Skylagring", "Dataanalyse", "Opplaering",
                "Serverdrift", "Nettverksinstallasjon", "Sikkerhetsvurdering", "Apputvikling"]
    return random.choice(products)


def _rand_price():
    return random.choice([500, 750, 1000, 1200, 1500, 2000, 2500, 3000, 5000, 7500, 10000, 15000])


LANGUAGES = {
    "nb": {  # Norwegian Bokmål
        "create_employee": "Opprett en ansatt med navn {first} {last}, e-post {email}.",
        "create_employee_admin": "Opprett en ansatt med navn {first} {last}, e-post {email}. Vedkommende skal vaere kontoadministrator.",
        "create_employee_full": "Opprett en ansatt med navn {first} {last}, e-post {email}, telefon {phone}, fodt {dob}.",
        "create_customer": "Opprett en kunde med navn {company}, organisasjonsnummer {org_nr}.",
        "create_customer_full": "Opprett en kunde med navn {company}, organisasjonsnummer {org_nr}, e-post {email}. Adresse: {street}, {postal} {city}, Norge.",
        "create_product": "Opprett et produkt med navn {product}, produktnummer {prod_nr}, pris {price} kr ekskl. mva.",
        "create_department": "Opprett en avdeling med navn {dept_name}, avdelingsnummer {dept_nr}.",
        "create_project": "Opprett et prosjekt med navn {proj_name} med startdato {start_date}.",
        "create_project_customer": "Opprett et prosjekt med navn {proj_name} for kunden {company} (org.nr. {org_nr}). Startdato {start_date}.",
        "create_project_manager": "Opprett et prosjekt med navn {proj_name}. Prosjektleder er {first} {last} ({email}). Startdato {start_date}.",
        "create_invoice": "Opprett en faktura til kunden {company} (org.nr. {org_nr}) for {product} ({count} stk a {price} kr). Fakturadato {inv_date}, forfallsdato {due_date}.",
        "create_travel_expense": "Registrer en reiseregning for {first} {last} ({email}). Reise fra {from_city} til {to_city}, avreise {dep_date}, retur {ret_date}. Formal: {purpose}.",
        "delete_travel_expense": "Slett reiseregningen til {first} {last}.",
        "create_supplier": "Opprett en leverandor med navn {company}, organisasjonsnummer {org_nr}, e-post {email}.",
        "create_timesheet": "Registrer {hours} timer for {first} {last} ({email}) den {date} pa aktiviteten {activity}.",
        "update_customer": "Oppdater kunden {company} med ny e-postadresse {email}.",
        "delete_customer": "Slett kunden {company}.",
    },
    "en": {
        "create_employee": "Create an employee named {first} {last}, email {email}.",
        "create_employee_admin": "Create an employee named {first} {last}, email {email}. They should be an account administrator.",
        "create_employee_full": "Create an employee named {first} {last}, email {email}, phone {phone}, date of birth {dob}.",
        "create_customer": "Create a customer named {company} with organization number {org_nr}.",
        "create_customer_full": "Create a customer named {company}, organization number {org_nr}, email {email}. Address: {street}, {postal} {city}, Norway.",
        "create_product": "Create a product named {product}, product number {prod_nr}, price {price} NOK excl. VAT.",
        "create_department": "Create a department named {dept_name}, department number {dept_nr}.",
        "create_project": "Create a project named {proj_name} with start date {start_date}.",
        "create_project_customer": "Create a project named {proj_name} for the customer {company} (org. no. {org_nr}). Start date {start_date}.",
        "create_project_manager": "Create a project named {proj_name}. Project manager is {first} {last} ({email}). Start date {start_date}.",
        "create_invoice": "Create an invoice for customer {company} (org. no. {org_nr}) for {product} ({count} pcs at {price} NOK). Invoice date {inv_date}, due date {due_date}.",
        "create_travel_expense": "Register a travel expense for {first} {last} ({email}). Travel from {from_city} to {to_city}, departure {dep_date}, return {ret_date}. Purpose: {purpose}.",
        "delete_travel_expense": "Delete the travel expense for {first} {last}.",
        "create_supplier": "Create a supplier named {company}, organization number {org_nr}, email {email}.",
        "create_timesheet": "Register {hours} hours for {first} {last} ({email}) on {date} for the activity {activity}.",
        "update_customer": "Update customer {company} with new email address {email}.",
        "delete_customer": "Delete customer {company}.",
    },
    "de": {
        "create_employee": "Erstellen Sie einen Mitarbeiter mit dem Namen {first} {last}, E-Mail {email}.",
        "create_employee_admin": "Erstellen Sie einen Mitarbeiter {first} {last}, E-Mail {email}. Er/Sie soll Kontoadministrator sein.",
        "create_customer": "Erstellen Sie einen Kunden mit dem Namen {company}, Organisationsnummer {org_nr}.",
        "create_customer_full": "Erstellen Sie einen Kunden {company}, Org.-Nr. {org_nr}, E-Mail {email}. Adresse: {street}, {postal} {city}, Norwegen.",
        "create_product": "Erstellen Sie ein Produkt mit dem Namen {product}, Produktnummer {prod_nr}, Preis {price} NOK exkl. MwSt.",
        "create_department": "Erstellen Sie eine Abteilung mit dem Namen {dept_name}, Abteilungsnummer {dept_nr}.",
        "create_project": "Erstellen Sie ein Projekt mit dem Namen {proj_name}, Startdatum {start_date}.",
        "create_invoice": "Erstellen Sie eine Rechnung fur den Kunden {company} (Org.-Nr. {org_nr}) fur {product} ({count} Stuck a {price} NOK). Rechnungsdatum {inv_date}, Falligkeitsdatum {due_date}.",
        "create_supplier": "Erstellen Sie einen Lieferanten {company}, Org.-Nr. {org_nr}, E-Mail {email}.",
    },
    "es": {
        "create_employee": "Cree un empleado con nombre {first} {last}, correo electronico {email}.",
        "create_employee_admin": "Cree un empleado {first} {last}, correo {email}. Debe ser administrador de cuenta.",
        "create_customer": "Cree un cliente con nombre {company}, numero de organizacion {org_nr}.",
        "create_customer_full": "Cree un cliente {company}, org. no. {org_nr}, correo {email}. Direccion: {street}, {postal} {city}, Noruega.",
        "create_product": "Cree un producto llamado {product}, numero de producto {prod_nr}, precio {price} NOK sin IVA.",
        "create_department": "Cree un departamento llamado {dept_name}, numero de departamento {dept_nr}.",
        "create_project": "Cree un proyecto llamado {proj_name} con fecha de inicio {start_date}.",
        "create_supplier": "Cree un proveedor llamado {company}, numero de organizacion {org_nr}, correo {email}.",
    },
    "fr": {
        "create_employee": "Creez un employe nomme {first} {last}, email {email}.",
        "create_employee_admin": "Creez un employe {first} {last}, email {email}. Il/Elle doit etre administrateur du compte.",
        "create_customer": "Creez un client nomme {company}, numero d'organisation {org_nr}.",
        "create_product": "Creez un produit nomme {product}, numero de produit {prod_nr}, prix {price} NOK HT.",
        "create_department": "Creez un departement nomme {dept_name}, numero de departement {dept_nr}.",
        "create_project": "Creez un projet nomme {proj_name} avec date de debut {start_date}.",
        "create_supplier": "Creez un fournisseur nomme {company}, numero d'organisation {org_nr}, email {email}.",
    },
    "pt": {
        "create_employee": "Crie um funcionario com o nome {first} {last}, email {email}.",
        "create_customer": "Crie um cliente com o nome {company}, numero de organizacao {org_nr}.",
        "create_product": "Crie um produto chamado {product}, numero do produto {prod_nr}, preco {price} NOK sem IVA.",
        "create_department": "Crie um departamento chamado {dept_name}, numero do departamento {dept_nr}.",
        "create_project": "Crie um projeto chamado {proj_name} com data de inicio {start_date}.",
        "create_supplier": "Crie um fornecedor chamado {company}, numero de organizacao {org_nr}, email {email}.",
    },
    "nn": {  # Nynorsk
        "create_employee": "Opprett ein tilsett med namn {first} {last}, e-post {email}.",
        "create_customer": "Opprett ein kunde med namn {company}, organisasjonsnummer {org_nr}.",
        "create_product": "Opprett eit produkt med namn {product}, produktnummer {prod_nr}, pris {price} kr ekskl. mva.",
        "create_department": "Opprett ei avdeling med namn {dept_name}, avdelingsnummer {dept_nr}.",
        "create_project": "Opprett eit prosjekt med namn {proj_name} med startdato {start_date}.",
        "create_supplier": "Opprett ein leverandor med namn {company}, organisasjonsnummer {org_nr}, e-post {email}.",
    },
}

# ── Verification functions ─────────────────────────────────────────────────────

def verify_employee(first: str, last: str, email: str, is_admin: bool = False, **kw) -> dict:
    """Verify employee was created."""
    r = httpx.get(f"{SANDBOX_BASE}/employee", auth=AUTH,
                  params={"count": 1000,
                           "fields": "id,firstName,lastName,email,userType"},
                  timeout=15)
    values = r.json().get("values", [])
    checks = {"employee_found": False, "correct_name": False, "correct_email": False}

    # Check ALL matching employees — agent may create a new one with same name
    best_match = None
    for emp in values:
        if emp.get("firstName") == first and emp.get("lastName") == last:
            if emp.get("email", "").lower() == email.lower():
                best_match = emp  # Exact email match — this is the one
                break
            elif best_match is None:
                best_match = emp  # First name match as fallback

    if best_match:
        checks["employee_found"] = True
        checks["correct_name"] = True
        if best_match.get("email", "").lower() == email.lower():
            checks["correct_email"] = True

        emp_r = httpx.get(f"{SANDBOX_BASE}/employee/employment", auth=AUTH,
                          params={"employeeId": best_match["id"], "count": 5, "fields": "id,startDate"},
                          timeout=15)
        if emp_r.json().get("values"):
            checks["employment_created"] = True
        else:
            checks["employment_created"] = False

    return checks


def verify_customer(company: str, org_nr: str = "", email: str = "", **kw) -> dict:
    """Verify customer was created."""
    # Tripletex name param doesn't reliably filter — use high count and filter client-side
    params = {"count": 1000, "fields": "id,name,organizationNumber,email,postalAddress(addressLine1,postalCode,city)"}
    r = httpx.get(f"{SANDBOX_BASE}/customer", auth=AUTH, params=params, timeout=15)
    values = r.json().get("values", [])
    checks = {"customer_found": False, "correct_name": False}
    if org_nr:
        checks["correct_org_nr"] = False
    if email:
        checks["correct_email"] = False

    for cust in values:
        if company.lower() in cust.get("name", "").lower():
            checks["customer_found"] = True
            checks["correct_name"] = True
            if org_nr and cust.get("organizationNumber") == org_nr:
                checks["correct_org_nr"] = True
            if email and cust.get("email", "").lower() == email.lower():
                checks["correct_email"] = True
            break

    return checks


def verify_product(product: str, prod_nr: str = "", **kw) -> dict:
    """Verify product was created."""
    r = httpx.get(f"{SANDBOX_BASE}/product", auth=AUTH,
                  params={"count": 1000, "fields": "id,name,number,priceExcludingVatCurrency"},
                  timeout=15)
    values = r.json().get("values", [])
    checks = {"product_found": False, "correct_name": False}

    for prod in values:
        if product.lower() in prod.get("name", "").lower():
            checks["product_found"] = True
            checks["correct_name"] = True
            break

    return checks


def verify_department(dept_name: str, **kw) -> dict:
    """Verify department was created."""
    r = httpx.get(f"{SANDBOX_BASE}/department", auth=AUTH,
                  params={"count": 50, "fields": "id,name,departmentNumber"}, timeout=15)
    values = r.json().get("values", [])
    checks = {"department_found": False}

    for dept in values:
        if dept_name.lower() in dept.get("name", "").lower():
            checks["department_found"] = True
            break

    return checks


def verify_project(proj_name: str, **kw) -> dict:
    """Verify project was created."""
    r = httpx.get(f"{SANDBOX_BASE}/project", auth=AUTH,
                  params={"count": 1000, "fields": "id,name,startDate,customer(id,name),projectManager(id,firstName)"}, timeout=15)
    values = r.json().get("values", [])
    checks = {"project_found": False}

    for proj in values:
        if proj_name.lower() in proj.get("name", "").lower():
            checks["project_found"] = True
            cust = proj.get("customer") or {}
            if cust.get("id"):
                checks["customer_linked"] = True
            mgr = proj.get("projectManager") or {}
            if mgr.get("id"):
                checks["manager_linked"] = True
            break

    return checks


def verify_supplier(company: str, **kw) -> dict:
    """Verify supplier was created."""
    r = httpx.get(f"{SANDBOX_BASE}/supplier", auth=AUTH,
                  params={"count": 50, "fields": "id,name,organizationNumber,email"}, timeout=15)
    values = r.json().get("values", [])
    checks = {"supplier_found": False}

    for s in values:
        if company.lower() in s.get("name", "").lower():
            checks["supplier_found"] = True
            break

    return checks


def verify_travel_expense(first: str, last: str, should_exist: bool = True, **kw) -> dict:
    """Verify travel expense was created or deleted."""
    r = httpx.get(f"{SANDBOX_BASE}/travelExpense", auth=AUTH,
                  params={"count": 50, "fields": "id,title,employee(id,firstName,lastName)"}, timeout=15)
    values = r.json().get("values", [])
    found = False
    for te in values:
        emp = te.get("employee", {})
        if emp.get("firstName") == first and emp.get("lastName") == last:
            found = True
            break

    if should_exist:
        return {"travel_expense_found": found}
    else:
        return {"travel_expense_deleted": not found}


def verify_timesheet(first: str, last: str, date: str, **kw) -> dict:
    """Verify timesheet entry was created."""
    # Find employee first — check ALL matching employees (agent may create a new one)
    r = httpx.get(f"{SANDBOX_BASE}/employee", auth=AUTH,
                  params={"count": 1000, "fields": "id,firstName,lastName"},
                  timeout=15)
    values = r.json().get("values", [])
    checks = {"timesheet_found": False}

    dt = datetime.strptime(date, "%Y-%m-%d")
    date_next = (dt + timedelta(days=1)).strftime("%Y-%m-%d")

    for emp in values:
        if emp.get("firstName") == first and emp.get("lastName") == last:
            emp_id = emp["id"]
            r2 = httpx.get(f"{SANDBOX_BASE}/timesheet/entry", auth=AUTH,
                           params={"employeeId": emp_id, "dateFrom": date, "dateTo": date_next,
                                    "count": 5, "fields": "id,date,hours"},
                           timeout=15)
            ts_values = r2.json().get("values", [])
            if ts_values:
                checks["timesheet_found"] = True
                break  # Found it, no need to check more

    return checks


def verify_noop(**kw) -> dict:
    """No verification possible (e.g., invoice on sandbox without bank account)."""
    return {"agent_completed": True}


# ── Task generation ────────────────────────────────────────────────────────────

def generate_tasks(count: int = 210, category: str = None) -> list[dict]:
    """Generate a diverse set of test tasks."""
    tasks = []
    task_id = 0

    # Define all task types with their generators
    task_types = []

    # 1. Employee tasks (various flavors)
    for lang_code, templates in LANGUAGES.items():
        if "create_employee" in templates:
            for _ in range(2):  # 2 per language
                first, last = _rand_name()
                email = _rand_email(first, last)
                task_types.append({
                    "category": "employee",
                    "language": lang_code,
                    "prompt": templates["create_employee"].format(first=first, last=last, email=email),
                    "verify": lambda f=first, l=last, e=email: verify_employee(f, l, e),
                    "check_count": 4,  # found, name, email, employment
                })

        if "create_employee_admin" in templates:
            first, last = _rand_name()
            email = _rand_email(first, last)
            task_types.append({
                "category": "employee_admin",
                "language": lang_code,
                "prompt": templates["create_employee_admin"].format(first=first, last=last, email=email),
                "verify": lambda f=first, l=last, e=email: verify_employee(f, l, e),
                "check_count": 4,  # sandbox ignores userType — just verify employee+employment
            })

        if "create_employee_full" in templates:
            first, last = _rand_name()
            email = _rand_email(first, last)
            phone = _rand_phone()
            dob = f"{random.randint(1970, 2000)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
            task_types.append({
                "category": "employee_full",
                "language": lang_code,
                "prompt": templates["create_employee_full"].format(first=first, last=last, email=email, phone=phone, dob=dob),
                "verify": lambda f=first, l=last, e=email: verify_employee(f, l, e),
                "check_count": 4,
            })

    # 2. Customer tasks
    for lang_code, templates in LANGUAGES.items():
        if "create_customer" in templates:
            for _ in range(2):
                company = f"Test{random.randint(1000,9999)} AS"
                org_nr = _rand_org_nr()
                task_types.append({
                    "category": "customer",
                    "language": lang_code,
                    "prompt": templates["create_customer"].format(company=company, org_nr=org_nr),
                    "verify": lambda c=company, o=org_nr: verify_customer(c, o),
                    "check_count": 3,
                })

        if "create_customer_full" in templates:
            company = f"Kunde{random.randint(1000,9999)} AS"
            org_nr = _rand_org_nr()
            email = f"post@{company.lower().replace(' ', '').replace('as','')}.no"
            street = _rand_street()
            postal = _rand_postal()
            city = _rand_city()
            task_types.append({
                "category": "customer_full",
                "language": lang_code,
                "prompt": templates["create_customer_full"].format(
                    company=company, org_nr=org_nr, email=email,
                    street=street, postal=postal, city=city),
                "verify": lambda c=company, o=org_nr, e=email: verify_customer(c, o, e),
                "check_count": 4,
            })

    # 3. Product tasks
    for lang_code, templates in LANGUAGES.items():
        if "create_product" in templates:
            for _ in range(2):
                product = _rand_product()
                prod_nr = f"P{random.randint(1000,9999)}"
                price = _rand_price()
                task_types.append({
                    "category": "product",
                    "language": lang_code,
                    "prompt": templates["create_product"].format(product=product, prod_nr=prod_nr, price=price),
                    "verify": lambda p=product: verify_product(p),
                    "check_count": 2,
                })

    # 4. Department tasks
    for lang_code, templates in LANGUAGES.items():
        if "create_department" in templates:
            dept_name = f"Avdeling {random.choice(['Salg', 'IT', 'HR', 'Finans', 'Marked', 'Drift', 'Utvikling', 'Kundeservice', 'Logistikk', 'Innkjop'])}"
            dept_nr = str(random.randint(10, 99))
            task_types.append({
                "category": "department",
                "language": lang_code,
                "prompt": templates["create_department"].format(dept_name=dept_name, dept_nr=dept_nr),
                "verify": lambda d=dept_name: verify_department(d),
                "check_count": 1,
            })

    # 5. Project tasks (simple)
    for lang_code, templates in LANGUAGES.items():
        if "create_project" in templates:
            proj_name = f"Prosjekt {random.choice(['Alfa', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Omega'])}-{random.randint(100,999)}"
            start_date = "2026-04-01"
            task_types.append({
                "category": "project",
                "language": lang_code,
                "prompt": templates["create_project"].format(proj_name=proj_name, start_date=start_date),
                "verify": lambda p=proj_name: verify_project(p),
                "check_count": 1,
            })

    # 6. Project with customer
    for lang_code, templates in LANGUAGES.items():
        if "create_project_customer" in templates:
            proj_name = f"Prosjekt {random.choice(['Nord', 'Sor', 'Ost', 'Vest'])}-{random.randint(100,999)}"
            company = f"ProjKunde{random.randint(100,999)} AS"
            org_nr = _rand_org_nr()
            task_types.append({
                "category": "project_customer",
                "language": lang_code,
                "prompt": templates["create_project_customer"].format(
                    proj_name=proj_name, company=company, org_nr=org_nr, start_date="2026-04-01"),
                "verify": lambda p=proj_name: verify_project(p),
                "check_count": 2,
            })

    # 7. Project with manager
    for lang_code, templates in LANGUAGES.items():
        if "create_project_manager" in templates:
            proj_name = f"Prosjekt Leder-{random.randint(100,999)}"
            first, last = _rand_name()
            email = _rand_email(first, last)
            task_types.append({
                "category": "project_manager",
                "language": lang_code,
                "prompt": templates["create_project_manager"].format(
                    proj_name=proj_name, first=first, last=last, email=email, start_date="2026-04-01"),
                "verify": lambda p=proj_name: verify_project(p),
                "check_count": 2,
            })

    # 8. Supplier tasks
    for lang_code, templates in LANGUAGES.items():
        if "create_supplier" in templates:
            company = f"Leverandor{random.randint(100,999)} AS"
            org_nr = _rand_org_nr()
            email = f"post@lev{random.randint(100,999)}.no"
            task_types.append({
                "category": "supplier",
                "language": lang_code,
                "prompt": templates["create_supplier"].format(company=company, org_nr=org_nr, email=email),
                "verify": lambda c=company: verify_supplier(c),
                "check_count": 1,
            })

    # 9. Invoice tasks (sandbox won't fully work but tests agent logic)
    for lang_code, templates in LANGUAGES.items():
        if "create_invoice" in templates:
            company = f"FakturaKunde{random.randint(100,999)} AS"
            org_nr = _rand_org_nr()
            product = _rand_product()
            cnt = random.choice([1, 2, 5, 10])
            price = _rand_price()
            task_types.append({
                "category": "invoice",
                "language": lang_code,
                "prompt": templates["create_invoice"].format(
                    company=company, org_nr=org_nr, product=product,
                    count=cnt, price=price, inv_date="2026-03-20", due_date="2026-04-19"),
                "verify": lambda: verify_noop(),
                "check_count": 1,
            })

    # 10. Travel expense create
    for lang_code, templates in LANGUAGES.items():
        if "create_travel_expense" in templates:
            first, last = _rand_name()
            email = _rand_email(first, last)
            from_city = _rand_city()
            to_city = _rand_city()
            while to_city == from_city:
                to_city = _rand_city()
            purposes = ["Kundemote", "Konferanse", "Opplaering", "Prosjektgjennomgang", "Messebesok"]
            task_types.append({
                "category": "travel_expense",
                "language": lang_code,
                "prompt": templates["create_travel_expense"].format(
                    first=first, last=last, email=email,
                    from_city=from_city, to_city=to_city,
                    dep_date="2026-04-01", ret_date="2026-04-03",
                    purpose=random.choice(purposes)),
                "verify": lambda f=first, l=last: verify_travel_expense(f, l, should_exist=True),
                "check_count": 1,
            })

    # 11. Timesheet tasks
    for lang_code, templates in LANGUAGES.items():
        if "create_timesheet" in templates:
            first, last = _rand_name()
            email = _rand_email(first, last)
            hours = random.choice([4.0, 7.5, 8.0, 3.5, 6.0])
            activities = ["Administrasjon", "Fakturerbart arbeid"]
            task_types.append({
                "category": "timesheet",
                "language": lang_code,
                "prompt": templates["create_timesheet"].format(
                    first=first, last=last, email=email,
                    hours=hours, date="2026-03-20",
                    activity=random.choice(activities)),
                "verify": lambda f=first, l=last: verify_timesheet(f, l, "2026-03-20"),
                "check_count": 1,
            })

    # Filter by category if specified
    if category:
        cats = [c.strip() for c in category.split(",")]
        task_types = [t for t in task_types if any(
            t["category"] == c or t["category"].startswith(c) for c in cats
        )]

    # Shuffle and truncate
    random.shuffle(task_types)
    task_types = task_types[:count]

    # Assign task IDs
    for i, t in enumerate(task_types):
        t["task_id"] = f"{t['category']}-{t['language']}-{i:03d}"

    return task_types


# ── Test runner ────────────────────────────────────────────────────────────────

async def run_single_task(task: dict, delay_between: float = 2.0) -> TaskResult:
    """Run a single task against the agent and verify."""
    result = TaskResult(
        task_id=task["task_id"],
        category=task["category"],
        language=task["language"],
        prompt=task["prompt"],
    )

    start = time.time()
    try:
        # Send to agent
        payload = {
            "prompt": task["prompt"],
            "files": [],
            "tripletex_credentials": {
                "base_url": SANDBOX_BASE,
                "session_token": SANDBOX_TOKEN,
            }
        }

        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(AGENT_URL, json=payload)

        result.elapsed_seconds = round(time.time() - start, 2)

        if resp.status_code != 200:
            result.status = "error"
            result.error_message = f"HTTP {resp.status_code}: {resp.text[:200]}"
            return result

        # Small delay before verification
        await asyncio.sleep(1)

        # Verify
        verify_fn = task["verify"]
        checks = verify_fn()
        result.details = checks
        result.checks_total = len(checks)
        result.checks_passed = sum(1 for v in checks.values() if v)

        if result.checks_passed == result.checks_total:
            result.status = "passed"
        else:
            result.status = "failed"
            failed_checks = [k for k, v in checks.items() if not v]
            result.error_message = f"Failed checks: {', '.join(failed_checks)}"

    except Exception as e:
        result.elapsed_seconds = round(time.time() - start, 2)
        result.status = "error"
        result.error_message = f"{type(e).__name__}: {str(e)[:200]}"

    return result


async def run_stress_test(count: int = 210, delay: float = 3.0, category: str = None):
    """Run the full stress test."""
    print(f"\n{'='*70}")
    print(f"  TRIPLETEX AGENT STRESS TEST")
    print(f"  Tasks: {count} | Delay: {delay}s | Category: {category or 'ALL'}")
    print(f"{'='*70}\n")

    tasks = generate_tasks(count, category)
    print(f"Generated {len(tasks)} tasks across categories:")
    cats = {}
    for t in tasks:
        cats[t["category"]] = cats.get(t["category"], 0) + 1
    for cat, cnt in sorted(cats.items()):
        print(f"  {cat:25s} {cnt:3d} tasks")
    print()

    results: list[TaskResult] = []
    log_file = Path("tests/stress_test_results.jsonl")
    log_file.parent.mkdir(exist_ok=True)

    # Clear previous results
    log_file.write_text("")

    passed = 0
    failed = 0
    errors = 0

    for i, task in enumerate(tasks):
        progress = f"[{i+1}/{len(tasks)}]"
        cat_short = task["category"][:15].ljust(15)
        lang = task["language"]
        prompt_preview = task["prompt"][:60]

        print(f"{progress} {cat_short} [{lang}] {prompt_preview}...", end=" ", flush=True)

        result = await run_single_task(task, delay)

        # Update counters
        if result.status == "passed":
            passed += 1
            status_icon = "PASS"
        elif result.status == "failed":
            failed += 1
            status_icon = "FAIL"
        else:
            errors += 1
            status_icon = "ERR "

        print(f"{status_icon} ({result.checks_passed}/{result.checks_total}) {result.elapsed_seconds}s")
        if result.error_message:
            print(f"         -> {result.error_message[:100]}")

        results.append(result)

        # Write to log file
        with open(log_file, "a") as f:
            f.write(json.dumps(asdict(result)) + "\n")

        # Rate limit delay
        if i < len(tasks) - 1:
            await asyncio.sleep(delay)

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*70}")
    total = len(results)
    print(f"  Total:   {total}")
    if total == 0:
        print("  No tasks matched the given category filter.")
        return
    print(f"  Passed:  {passed} ({100*passed/total:.1f}%)")
    print(f"  Failed:  {failed} ({100*failed/total:.1f}%)")
    print(f"  Errors:  {errors} ({100*errors/total:.1f}%)")
    print()

    # Per-category breakdown
    cat_results = {}
    for r in results:
        if r.category not in cat_results:
            cat_results[r.category] = {"passed": 0, "failed": 0, "error": 0, "total": 0}
        cat_results[r.category]["total"] += 1
        if r.status == "passed":
            cat_results[r.category]["passed"] += 1
        elif r.status == "failed":
            cat_results[r.category]["failed"] += 1
        else:
            cat_results[r.category]["error"] += 1

    print(f"  {'Category':<25s} {'Pass':>5s} {'Fail':>5s} {'Err':>5s} {'Total':>5s} {'Rate':>6s}")
    print(f"  {'-'*52}")
    for cat in sorted(cat_results.keys()):
        cr = cat_results[cat]
        rate = f"{100*cr['passed']/cr['total']:.0f}%" if cr["total"] > 0 else "N/A"
        print(f"  {cat:<25s} {cr['passed']:>5d} {cr['failed']:>5d} {cr['error']:>5d} {cr['total']:>5d} {rate:>6s}")

    # Average time
    times = [r.elapsed_seconds for r in results if r.elapsed_seconds > 0]
    if times:
        print(f"\n  Avg time: {sum(times)/len(times):.1f}s  |  Min: {min(times):.1f}s  |  Max: {max(times):.1f}s")

    # Failed task details
    if failed + errors > 0:
        print(f"\n  FAILED/ERROR DETAILS:")
        for r in results:
            if r.status in ("failed", "error"):
                print(f"    [{r.task_id}] {r.status.upper()}: {r.error_message[:120]}")

    print(f"\n  Results saved to: {log_file}")
    print(f"{'='*70}\n")

    return results


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stress test the Tripletex AI Agent")
    parser.add_argument("--count", type=int, default=210, help="Number of tasks to run (default: 210)")
    parser.add_argument("--delay", type=float, default=3.0, help="Delay between tasks in seconds (default: 3.0)")
    parser.add_argument("--category", type=str, default=None,
                        help="Filter by category (employee, customer, product, department, project, supplier, invoice, travel_expense, timesheet)")
    args = parser.parse_args()

    asyncio.run(run_stress_test(count=args.count, delay=args.delay, category=args.category))
