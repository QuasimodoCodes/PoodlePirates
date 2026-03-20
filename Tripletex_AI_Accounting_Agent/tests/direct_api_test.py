"""Direct API test — validates all task types work correctly without Gemini."""
import httpx
import json
import os
import sys

BASE = os.environ.get("SANDBOX_BASE", "https://kkpqfuj-amager.tripletex.dev/v2")
TOKEN = os.environ.get("SANDBOX_TOKEN", "")
AUTH = ("0", TOKEN)

results = []

def test(name, method, path, body=None, params=None, expected_status=201):
    """Run a single API test."""
    url = f"{BASE}{path}"
    kwargs = {"auth": AUTH, "timeout": 15}
    if body is not None:
        kwargs["json"] = body
    if params is not None:
        kwargs["params"] = params
    
    if method == "POST":
        r = httpx.post(url, **kwargs)
    elif method == "GET":
        r = httpx.get(url, **kwargs)
        expected_status = 200
    elif method == "PUT":
        r = httpx.put(url, **kwargs)
    elif method == "DELETE":
        r = httpx.delete(url, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    ok = r.status_code == expected_status
    data = r.json() if r.status_code < 500 else {}
    value = data.get("value", {})
    
    status_icon = "PASS" if ok else "FAIL"
    print(f"  [{status_icon}] {method} {path} -> {r.status_code}")
    if not ok:
        # Show error details
        msgs = data.get("validationMessages", [])
        if msgs:
            for m in msgs[:3]:
                print(f"       Validation: {m.get('field','?')}: {m.get('message','?')}")
        else:
            err = data.get("error", {})
            if isinstance(err, dict):
                print(f"       Error: {err.get('error', str(data)[:200])}")
            else:
                print(f"       Response: {str(data)[:200]}")
    
    results.append((name, ok, r.status_code))
    return value, ok


print("=" * 60)
print("  DIRECT API TEST — No Gemini, pure API validation")
print("=" * 60)

# 1. Employee (requires email, userType, dateOfBirth; may require department)
print("\n--- EMPLOYEE ---")
# Get a department ID first (some accounts require it)
dept_r = httpx.get(f"{BASE}/department", auth=AUTH, params={"count": 1, "fields": "id"}, timeout=15)
dept_vals = dept_r.json().get("values", [])
dept_id = dept_vals[0]["id"] if dept_vals else None

emp_body = {
    "firstName": "ApiTest2",
    "lastName": "Employee2",
    "email": "apitest2@test.no",
    "userType": "STANDARD",
    "dateOfBirth": "1990-01-15"
}
if dept_id:
    emp_body["department"] = {"id": dept_id}

emp, ok = test("employee_create", "POST", "/employee", emp_body)
emp_id = emp.get("id")
if emp_id:
    test("employment_create", "POST", "/employee/employment", {
        "employee": {"id": emp_id},
        "startDate": "2025-01-01",
        "isMainEmployer": True
    })

# 2. Customer with email
print("\n--- CUSTOMER ---")
cust, ok = test("customer_create", "POST", "/customer", {
    "name": "ApiTestKunde AS",
    "organizationNumber": "912345001",
    "email": "info@apitestkunde.no",
    "postalAddress": {
        "addressLine1": "Testgata 10",
        "postalCode": "0180",
        "city": "Oslo",
        "country": {"id": 161}
    }
})
if ok:
    # Verify email was set correctly
    cust_email = cust.get("email", "")
    inv_email = cust.get("invoiceEmail", "")
    if cust_email == "info@apitestkunde.no":
        print(f"       email field set correctly: {cust_email}")
    else:
        print(f"       WARNING: email={cust_email}, invoiceEmail={inv_email}")

# 3. Product (omit vatType — some accounts reject certain VAT types)
print("\n--- PRODUCT ---")
prod, ok = test("product_create", "POST", "/product", {
    "name": "ApiTestProdukt2",
    "number": "ATP002",
    "priceExcludingVatCurrency": 750.0
})

# 4. Supplier
print("\n--- SUPPLIER ---")
supp, ok = test("supplier_create", "POST", "/supplier", {
    "name": "ApiTestLeverandor AS",
    "organizationNumber": "912345002",
    "email": "info@apitestlev.no",
    "postalAddress": {
        "addressLine1": "Leverandorgata 5",
        "postalCode": "5003",
        "city": "Bergen",
        "country": {"id": 161}
    }
})

# 5. Department
print("\n--- DEPARTMENT ---")
dept, ok = test("department_create", "POST", "/department", {
    "name": "ApiTestAvdeling",
    "departmentNumber": "ATD01"
})

# 6. Project (needs projectManager — use first employee in account)
print("\n--- PROJECT ---")
pm_r = httpx.get(f"{BASE}/employee", auth=AUTH, params={"count": 1, "fields": "id"}, timeout=15)
pm_id = pm_r.json().get("values", [{}])[0].get("id")
proj, ok = test("project_create", "POST", "/project", {
    "name": "ApiTestProsjekt2",
    "startDate": "2025-01-01",
    "projectManager": {"id": pm_id} if pm_id else None
})
proj_id = proj.get("id")

# 7. Travel Expense
print("\n--- TRAVEL EXPENSE ---")
if emp_id:
    te, ok = test("travel_expense_create", "POST", "/travelExpense", {
        "employee": {"id": emp_id},
        "title": "API Test Travel",
        "travelDetails": {
            "departureDate": "2025-03-01",
            "returnDate": "2025-03-03",
            "departureFrom": "Oslo",
            "destination": "Bergen",
            "purpose": "API Testing"
        }
    })
else:
    print("  [SKIP] No employee created — cannot test travel expense")

# 8. Timesheet Entry (need activity)
print("\n--- TIMESHEET ---")
if emp_id:
    # Find or create activity
    r = httpx.get(f"{BASE}/activity", auth=AUTH, params={"count": 5, "fields": "id,name"}, timeout=15)
    activities = r.json().get("values", [])
    act_id = activities[0]["id"] if activities else None
    
    if not act_id:
        act_resp, _ = test("activity_create", "POST", "/activity", {"name": "ApiTestActivity"})
        act_id = act_resp.get("id")
    else:
        print(f"  [INFO] Using existing activity: id={act_id}, name={activities[0].get('name')}")
    
    if act_id:
        ts, ok = test("timesheet_create", "POST", "/timesheet/entry", {
            "employee": {"id": emp_id},
            "date": "2025-03-15",
            "hours": 7.5,
            "activity": {"id": act_id}
        })
else:
    print("  [SKIP] No employee — cannot test timesheet")

# 9. Voucher (manual bookkeeping)
print("\n--- VOUCHER ---")
# Get account IDs for 1920 (Bank) and 3000 (Sales)
r = httpx.get(f"{BASE}/ledger/account", auth=AUTH,
              params={"number": "1920", "count": 1, "fields": "id,number,name"}, timeout=15)
bank_acc = r.json().get("values", [{}])[0] if r.json().get("values") else {}
r = httpx.get(f"{BASE}/ledger/account", auth=AUTH,
              params={"number": "3000", "count": 1, "fields": "id,number,name"}, timeout=15)
sales_acc = r.json().get("values", [{}])[0] if r.json().get("values") else {}

if bank_acc.get("id") and sales_acc.get("id"):
    print(f"  [INFO] Bank account: id={bank_acc['id']} ({bank_acc.get('name','')})")
    print(f"  [INFO] Sales account: id={sales_acc['id']} ({sales_acc.get('name','')})")
    voucher, ok = test("voucher_create", "POST", "/ledger/voucher", {
        "date": "2025-03-15",
        "description": "API Test Voucher v2",
        "postings": [
            {"row": 1, "date": "2025-03-15", "account": {"id": bank_acc["id"]}, "amount": 1000.0, "amountGross": 1000.0, "amountGrossCurrency": 1000.0, "currency": {"id": 1}},
            {"row": 2, "date": "2025-03-15", "account": {"id": sales_acc["id"]}, "amount": -1000.0, "amountGross": -1000.0, "amountGrossCurrency": -1000.0, "currency": {"id": 1}, "vatType": {"id": 3}}
        ]
    })
else:
    print("  [SKIP] Could not find bank/sales accounts")

# Summary
print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
print(f"  Passed: {passed}/{len(results)}")
print(f"  Failed: {failed}/{len(results)}")
for name, ok, status in results:
    icon = "PASS" if ok else "FAIL"
    print(f"    [{icon}] {name} (HTTP {status})")

if failed > 0:
    sys.exit(1)
