## 2024-06-25 - Python Embedded HTML A11y
**Learning:** Standard UI accessibility linters (like ESLint plugin for JSX/HTML) do not scan Python string literals used by lightweight backends like FastAPI's `HTMLResponse`. Accessibility attributes like `for`/`id` bindings are frequently missed in these templates.
**Action:** Always manually check `<label>` and `<input>` bindings, along with other ARIA properties, when editing HTML strings inside backend `.py` files.
