## 2024-05-18 - Reflected XSS in Admin Dashboard HTML Generation
**Vulnerability:** The admin dashboard manually generates HTML strings and was embedding unsanitized user query parameters (`msg`, `contains`) directly into attributes and elements, allowing for Reflected Cross-Site Scripting (XSS).
**Learning:** In applications that construct raw HTML directly (like `FastAPI`'s `HTMLResponse` combined with Python f-strings) rather than using a templating engine (like Jinja2) with auto-escaping, every user-controlled input must be manually sanitized.
**Prevention:** Always use `html.escape()` when embedding user input (such as query strings or form data) into raw HTML strings to prevent XSS attacks.
## $(date +%Y-%m-%d) - Parameterized Limit Queries in SQLite
**Vulnerability:** A `LIMIT` parameter was dynamically injected into a SQL query via f-strings (`f"{query} LIMIT {int(limit)}"`) in `PersistentStateDB.load_fills`. While `int(limit)` casting mitigates immediate arbitrary SQL injection in Python, it relies on application-level type enforcement rather than driver-level parameterization.
**Learning:** Even when inputs are cast to safe types like `int`, standard security best practices demand using parameterized queries (`LIMIT ?`) for all dynamic SQL components to implement defense in depth and future-proof the codebase against changes to the type coercion logic.
**Prevention:** Always use `sqlite3` placeholder parameterization (the `?` syntax) for `LIMIT`, `OFFSET`, and `ORDER BY` values (where supported by the driver) instead of Python string formatting.
