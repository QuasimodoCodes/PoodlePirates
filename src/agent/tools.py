"""
Tool definitions passed to Claude's tool-use API.

We use four generic HTTP tools (GET/POST/PUT/DELETE) rather than
one-per-endpoint so that Claude can reach any Tripletex API path
without us having to enumerate all 30 task types.
"""

TOOLS = [
    {
        "name": "tripletex_get",
        "description": (
            "Call a Tripletex API GET endpoint to read or search for data. "
            "Use this to look up IDs before creating resources, or to verify results."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "API path, e.g. '/employee' or '/customer/42'. Do NOT include the base URL.",
                },
                "params": {
                    "type": "object",
                    "description": "Optional query parameters as key-value pairs, e.g. {\"name\": \"Acme\", \"count\": 5}.",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "tripletex_post",
        "description": (
            "Call a Tripletex API POST endpoint to create a new resource. "
            "Returns the created object including its new id."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "API path, e.g. '/employee' or '/invoice'.",
                },
                "body": {
                    "type": "object",
                    "description": "JSON body for the resource to create.",
                },
            },
            "required": ["path", "body"],
        },
    },
    {
        "name": "tripletex_put",
        "description": (
            "Call a Tripletex API PUT endpoint to update an existing resource. "
            "Include the resource id in the path, e.g. '/employee/42'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "API path including id, e.g. '/employee/42'.",
                },
                "body": {
                    "type": "object",
                    "description": "Full updated resource body (Tripletex uses PUT not PATCH).",
                },
                "params": {
                    "type": "object",
                    "description": "Optional query parameters.",
                },
            },
            "required": ["path", "body"],
        },
    },
    {
        "name": "tripletex_delete",
        "description": (
            "Call a Tripletex API DELETE endpoint to remove a resource. "
            "Include the resource id in the path, e.g. '/employee/42'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "API path including id, e.g. '/travelExpense/99'.",
                },
            },
            "required": ["path"],
        },
    },
]
