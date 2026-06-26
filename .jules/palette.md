## 2024-05-18 - Missing Accessibility Bindings in Python String HTML
**Learning:** When generating HTML strings from a Python backend (like FastAPI HTMLResponse), a11y linters won't catch missing `for` attributes on `<label>` elements or missing `id` attributes on `<input>` elements. Form fields in the admin dashboard had no explicit association.
**Action:** Manually verify accessibility bindings (like `for`/`id` pairs) in HTML literal strings within Python backend code.
