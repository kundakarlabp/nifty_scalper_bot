## 2024-05-14 - Python String Literal Accessibility
**Learning:** This app generates its HTML directly via Python string literals (FastAPI `HTMLResponse` in `admin_dashboard.py`). Standard UI accessibility linters do not scan these string literals, so it's easy to miss basic accessibility features like `for`/`id` bindings on `<label>` and `<input>`.
**Action:** Always manually verify structural accessibility (like `for` and `id` linking for labels) when modifying the raw HTML strings in this backend file.
