## 2024-05-18 - Reflected XSS in Admin Dashboard HTML Generation
**Vulnerability:** The admin dashboard manually generates HTML strings and was embedding unsanitized user query parameters (`msg`, `contains`) directly into attributes and elements, allowing for Reflected Cross-Site Scripting (XSS).
**Learning:** In applications that construct raw HTML directly (like `FastAPI`'s `HTMLResponse` combined with Python f-strings) rather than using a templating engine (like Jinja2) with auto-escaping, every user-controlled input must be manually sanitized.
**Prevention:** Always use `html.escape()` when embedding user input (such as query strings or form data) into raw HTML strings to prevent XSS attacks.
## 2026-06-23 - Parameterize SQL LIMIT clause in PersistentStateDB
**Vulnerability:** The `load_fills` method in `PersistentStateDB` appended the `LIMIT` clause using string formatting (`f"{query} LIMIT {int(limit)}"`).
**Learning:** Even though the variable was cast to `int`, using string formatting to construct SQL queries is an anti-pattern that can trip static analysis tools and introduce risks if refactored poorly later.
**Prevention:** Always use parameterized queries (with the `?` placeholder in SQLite) for any dynamic part of an SQL query, including `LIMIT` clauses, to strictly separate code from data.
