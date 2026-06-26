## 2024-05-18 - Reflected XSS in Admin Dashboard HTML Generation
**Vulnerability:** The admin dashboard manually generates HTML strings and was embedding unsanitized user query parameters (`msg`, `contains`) directly into attributes and elements, allowing for Reflected Cross-Site Scripting (XSS).
**Learning:** In applications that construct raw HTML directly (like `FastAPI`'s `HTMLResponse` combined with Python f-strings) rather than using a templating engine (like Jinja2) with auto-escaping, every user-controlled input must be manually sanitized.
**Prevention:** Always use `html.escape()` when embedding user input (such as query strings or form data) into raw HTML strings to prevent XSS attacks.
## 2024-05-18 - SQL Injection in LIMIT clause
**Vulnerability:** The `load_fills` function in `src/nifty_scalper_bot/data/persistent_state.py` was appending the `limit` parameter directly into the SQL query string using an f-string (e.g., `query = f"{query} LIMIT {int(limit)}"`).
**Learning:** Even if a variable is type-cast to an integer before being inserted into an SQL string, it's a security anti-pattern. While Python's `int()` casting prevents actual SQL injection in this specific case, it violates the core security principle of always using parameterized queries for dynamic SQL components.
**Prevention:** Always prioritize parameterized queries (using the `?` placeholder) over string formatting (`f-strings`) in SQL statements to prevent SQL injection, regardless of type-casting or perceived safety of the input variable.
