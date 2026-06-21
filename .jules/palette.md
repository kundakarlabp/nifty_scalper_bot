## 2024-05-18 - HTML String Templates Accessibility
**Learning:** When UI HTML is rendered inside Python strings (like FastAPI `HTMLResponse`), standard frontend a11y linters (like eslint-plugin-jsx-a11y) do not scan them, allowing severe accessibility issues like unassociated labels (`<label>` without `for` matching an `<input id>`) to go undetected.
**Action:** Always manually verify label bindings (`for` and `id`) and ARIA attributes when modifying or adding string-based HTML templates in Python backends.
