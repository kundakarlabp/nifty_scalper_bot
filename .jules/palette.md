## 2024-05-24 - HTML Accessibility in Python String Templates
**Learning:** Standard UI accessibility linters (like eslint-plugin-jsx-a11y) do not scan Python string literals used for returning raw HTML (e.g., FastAPI HTMLResponse). As a result, critical a11y bindings like `for` and `id` attributes on form elements can easily be missed.
**Action:** When updating or reviewing Python files that generate HTML strings, manually check for proper semantic HTML and accessibility attributes, particularly form labels and focus management.
