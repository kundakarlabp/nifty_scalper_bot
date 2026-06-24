## 2024-05-18 - Reflected XSS in Admin Dashboard HTML Generation
**Vulnerability:** The admin dashboard manually generates HTML strings and was embedding unsanitized user query parameters (`msg`, `contains`) directly into attributes and elements, allowing for Reflected Cross-Site Scripting (XSS).
**Learning:** In applications that construct raw HTML directly (like `FastAPI`'s `HTMLResponse` combined with Python f-strings) rather than using a templating engine (like Jinja2) with auto-escaping, every user-controlled input must be manually sanitized.
**Prevention:** Always use `html.escape()` when embedding user input (such as query strings or form data) into raw HTML strings to prevent XSS attacks.
## 2024-05-18 - Parameterized Queries for Limit Clauses
**Vulnerability:** The application used string formatting (`f"{query} LIMIT {int(limit)}"`) instead of parameterized queries for a SQL LIMIT clause in the SQLite persistence module.
**Learning:** Even if type casting (like `int()`) prevents actual SQL injection by raising an error on malicious strings, utilizing parameterized queries (`LIMIT ?`) is a strict security best practice and establishes a defense-in-depth pattern across the entire query space.
**Prevention:** Always prioritize parameterized queries over string formatting in SQL statements to prevent SQL injection, regardless of type-casting or perceived safety of the input variable.
