## 2024-05-18 - Reflected XSS in Admin Dashboard HTML Generation
**Vulnerability:** The admin dashboard manually generates HTML strings and was embedding unsanitized user query parameters (`msg`, `contains`) directly into attributes and elements, allowing for Reflected Cross-Site Scripting (XSS).
**Learning:** In applications that construct raw HTML directly (like `FastAPI`'s `HTMLResponse` combined with Python f-strings) rather than using a templating engine (like Jinja2) with auto-escaping, every user-controlled input must be manually sanitized.
**Prevention:** Always use `html.escape()` when embedding user input (such as query strings or form data) into raw HTML strings to prevent XSS attacks.
## 2025-02-14 - Reflected XSS in Admin Dashboard Settings Input
**Vulnerability:** The admin dashboard reflected unsanitized configuration values (potentially originating from user input or modified environment files) directly into HTML input values during rendering.
**Learning:** Even internal configuration values or variables read from '.env' files should be treated as untrusted input when rendering HTML manually via f-strings, as they can contain characters that break the HTML context.
**Prevention:** Always use `html.escape()` for any dynamic variable interpolated into raw HTML string templates, particularly within HTML attributes, to prevent XSS.
