"""
Sandbox test suite — tests the full agent against each task type.

Usage:
    1. Set your sandbox token:
       set TRIPLETEX_SESSION_TOKEN=your_token_here

    2. Run all tests:
       python tests/test_sandbox.py

    3. Run a single test:
       python tests/test_sandbox.py customer

The sandbox is persistent, so created resources stay. Each test verifies
the agent's work by querying the API to check results.
"""
import os
import sys
import time
import httpx

BASE_URL = "https://kkpqfuj-amager.tripletex.dev/v2"
SESSION_TOKEN = os.getenv("TRIPLETEX_SESSION_TOKEN", "")
SERVER_URL = "http://localhost:8000"

if not SESSION_TOKEN:
    print("ERROR: Set TRIPLETEX_SESSION_TOKEN environment variable first!")
    print('  PowerShell: $env:TRIPLETEX_SESSION_TOKEN = "your_token"')
    sys.exit(1)

AUTH = ("0", SESSION_TOKEN)


def tripletex_get(path, params=None):
    """Direct Tripletex API call for verification."""
    resp = httpx.get(f"{BASE_URL}{path}", auth=AUTH, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def send_task(prompt, files=None):
    """Send a task to the local agent server."""
    payload = {
        "prompt": prompt,
        "files": files or [],
        "tripletex_credentials": {
            "base_url": BASE_URL,
            "session_token": SESSION_TOKEN,
        },
    }
    print(f"\n{'='*60}")
    print(f"TASK: {prompt[:100]}...")
    print(f"{'='*60}")
    start = time.time()
    resp = httpx.post(f"{SERVER_URL}/solve", json=payload, timeout=300)
    elapsed = round(time.time() - start, 1)
    print(f"  Response: {resp.json()} ({elapsed}s)")
    return resp.json()


# ── Test functions ────────────────────────────────────────────────────────────

def test_employee():
    """Test: Create employee with employment."""
    ts = int(time.time()) % 10000
    prompt = (
        f"Opprett en ansatt med fornavn TestEmp{ts} og etternavn Testesen. "
        f"E-post: testemp{ts}@example.org. Startdato: 2026-03-20."
    )
    send_task(prompt)

    # Verify
    result = tripletex_get("/employee", {"firstName": f"TestEmp{ts}", "count": 5})
    employees = result.get("values", [])
    if employees:
        emp = employees[0]
        print(f"  ✅ Employee found: {emp.get('firstName')} {emp.get('lastName')}")
        print(f"     Email: {emp.get('email')}")
        # Check employment
        emp_id = emp["id"]
        emps = tripletex_get("/employee/employment", {"employeeId": emp_id, "count": 5})
        if emps.get("values"):
            print(f"  ✅ Employment found: startDate={emps['values'][0].get('startDate')}")
        else:
            print(f"  ❌ No employment created!")
    else:
        print(f"  ❌ Employee not found!")


def test_customer():
    """Test: Create customer with address."""
    ts = int(time.time()) % 10000
    prompt = (
        f"Opprett kunden TestKunde{ts} AS med organisasjonsnummer 987654321. "
        f"Adressen er Testveien 42, 0150 Oslo. E-post: post@testkunde{ts}.no."
    )
    send_task(prompt)

    result = tripletex_get("/customer", {"name": f"TestKunde{ts}", "count": 5})
    customers = result.get("values", [])
    if customers:
        c = customers[0]
        print(f"  ✅ Customer found: {c.get('name')}")
        print(f"     Org: {c.get('organizationNumber')}, Email: {c.get('email')}")
    else:
        print(f"  ❌ Customer not found!")


def test_product():
    """Test: Create product with VAT."""
    ts = int(time.time()) % 10000
    prompt = (
        f"Create the product \"TestProduct{ts}\" with product number {ts}. "
        f"The price is 15000 NOK excluding VAT, using the standard rate of 25%."
    )
    send_task(prompt)

    result = tripletex_get("/product", {"name": f"TestProduct{ts}", "count": 5})
    products = result.get("values", [])
    if products:
        p = products[0]
        print(f"  ✅ Product found: {p.get('name')}")
        print(f"     Price: {p.get('priceExcludingVatCurrency')}, Number: {p.get('number')}")
    else:
        print(f"  ❌ Product not found!")


def test_project():
    """Test: Create project with customer and project manager."""
    ts = int(time.time()) % 10000
    prompt = (
        f"Crie o projeto \"TestProject{ts}\" vinculado ao cliente TestProjCust{ts} Lda "
        f"(org. nº 986713344). O gerente de projeto é TestMgr{ts} Silva "
        f"(testmgr{ts}@example.org)."
    )
    send_task(prompt)

    result = tripletex_get("/project", {"name": f"TestProject{ts}", "count": 5})
    projects = result.get("values", [])
    if projects:
        p = projects[0]
        print(f"  ✅ Project found: {p.get('name')}")
        print(f"     Start: {p.get('startDate')}")
    else:
        print(f"  ❌ Project not found!")


def test_department():
    """Test: Create department."""
    ts = int(time.time()) % 10000
    prompt = f"Erstelle die Abteilung \"TestDept{ts}\" mit der Abteilungsnummer {ts}."
    send_task(prompt)

    result = tripletex_get("/department", {"name": f"TestDept{ts}", "count": 5})
    depts = result.get("values", [])
    if depts:
        d = depts[0]
        print(f"  ✅ Department found: {d.get('name')}, number: {d.get('departmentNumber')}")
    else:
        print(f"  ❌ Department not found!")


def test_invoice():
    """Test: Create invoice for a customer."""
    ts = int(time.time()) % 10000
    prompt = (
        f"Create and send an invoice to customer InvCust{ts} Ltd (org no. 985190631) "
        f"for 25000 NOK excluding VAT. The invoice is for 'Consulting services'. "
        f"Use standard 25% VAT rate."
    )
    send_task(prompt)

    # Check if any invoice was created recently
    result = tripletex_get("/invoice", {"count": 5})
    invoices = result.get("values", [])
    if invoices:
        inv = invoices[-1]  # most recent
        print(f"  ✅ Invoice found: id={inv.get('id')}, amount={inv.get('amount')}")
    else:
        print(f"  ❌ No invoices found!")


# ── Main ──────────────────────────────────────────────────────────────────────

TESTS = {
    "employee": test_employee,
    "customer": test_customer,
    "product": test_product,
    "project": test_project,
    "department": test_department,
    "invoice": test_invoice,
}


def main():
    # Check server is running
    try:
        httpx.get(f"{SERVER_URL}/health", timeout=5)
    except Exception:
        print("ERROR: Server not running! Start it with: python main.py")
        sys.exit(1)

    # Check sandbox token works
    try:
        tripletex_get("/employee", {"count": 1})
        print("✅ Sandbox connection OK")
    except Exception as e:
        print(f"ERROR: Sandbox token invalid: {e}")
        sys.exit(1)

    # Run tests
    if len(sys.argv) > 1:
        name = sys.argv[1]
        if name in TESTS:
            TESTS[name]()
        else:
            print(f"Unknown test: {name}. Available: {', '.join(TESTS.keys())}")
    else:
        print(f"\nRunning all {len(TESTS)} tests...\n")
        for name, fn in TESTS.items():
            try:
                fn()
            except Exception as e:
                print(f"  ❌ {name} CRASHED: {e}")
            print()

    print("\n" + "=" * 60)
    print("Done! Check results above.")


if __name__ == "__main__":
    main()
