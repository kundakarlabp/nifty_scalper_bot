## 2024-05-18 - Reflected XSS in Admin Dashboard HTML Generation
**Vulnerability:** The admin dashboard manually generates HTML strings and was embedding unsanitized user query parameters (`msg`, `contains`) directly into attributes and elements, allowing for Reflected Cross-Site Scripting (XSS).
**Learning:** In applications that construct raw HTML directly (like `FastAPI`'s `HTMLResponse` combined with Python f-strings) rather than using a templating engine (like Jinja2) with auto-escaping, every user-controlled input must be manually sanitized.
**Prevention:** Always use `html.escape()` when embedding user input (such as query strings or form data) into raw HTML strings to prevent XSS attacks.
## 2024-05-18 - Parameterized LIMIT Query in SQLite
**Vulnerability:** A SQL query in `PersistentStateDB.load_fills` was constructing its `LIMIT` clause using string interpolation (`f"{query} LIMIT {int(limit)}"`). Although it cast the parameter to an `int` (mitigating immediate exploitation), it violates strict parameterization principles and could set a dangerous precedent.
**Learning:** Even when input appears safe (e.g., cast to integer), always use proper database parameterization (e.g., `?` placeholders) for all dynamic values, including `LIMIT` or `OFFSET` clauses in SQLite, to ensure defense-in-depth and avoid potential future injection vectors if the type-casting is ever modified.
**Prevention:** Strictly enforce the use of parameterized queries for all dynamically injected values in SQL statements.
