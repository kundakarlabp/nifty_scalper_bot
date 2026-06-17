## 2025-03-05 - Direct HTML Rendering in Backend
**Learning:** This application does not have a separate frontend repository or package.json. The UI (Admin Dashboard) is generated entirely via string templates within the Python backend using FastAPI `HTMLResponse`.
**Action:** When asked to make UX/UI enhancements, search for python files containing HTML formatting strings (like `admin_dashboard.py`) instead of looking for React/Vue components or node package managers. Apply UX improvements (like a11y attributes) directly into these python string templates.
