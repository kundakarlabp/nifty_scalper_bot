## 2024-05-19 - Reflected XSS in Admin Dashboard Flash Messages
**Vulnerability:** Reflected Cross-Site Scripting (XSS) in `src/nifty_scalper_bot/admin_dashboard.py` via the `msg` query parameter.
**Learning:** The dashboard renders raw HTML strings for responses instead of using a templating engine (like Jinja2) that auto-escapes output. This architectural choice makes the application highly susceptible to XSS anywhere user input is reflected in the HTML.
**Prevention:** Whenever rendering HTML via direct string interpolation, manually wrap all user-provided data (query parameters, form inputs) in `html.escape()` from the Python standard library before insertion.
