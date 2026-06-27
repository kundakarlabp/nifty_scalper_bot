## 2024-05-18 - Reflected XSS in Admin Dashboard HTML Generation
**Vulnerability:** The admin dashboard manually generates HTML strings and was embedding unsanitized user query parameters (`msg`, `contains`) directly into attributes and elements, allowing for Reflected Cross-Site Scripting (XSS).
**Learning:** In applications that construct raw HTML directly (like `FastAPI`'s `HTMLResponse` combined with Python f-strings) rather than using a templating engine (like Jinja2) with auto-escaping, every user-controlled input must be manually sanitized.
**Prevention:** Always use `html.escape()` when embedding user input (such as query strings or form data) into raw HTML strings to prevent XSS attacks.

## 2026-06-27 - SQL Injection Risk in LIMIT clause
**Vulnerability:** A string-formatted `LIMIT {int(limit)}` was used in `PersistentStateDB.load_fills`, allowing for potential SQL injection despite the type-casting if user-controlled input somehow bypassed validation.
**Learning:** Even when inputs are type-casted (e.g. `int()`), it is a security best practice to prioritize parameterized queries over string formatting (f-strings) in SQL statements for all dynamic SQL components, including `LIMIT` clauses, to ensure robust protection against SQL injection.
**Prevention:** Always use parameterized queries (with the `?` placeholder in SQLite) for all dynamic SQL components, regardless of the perceived safety of the input variable.
