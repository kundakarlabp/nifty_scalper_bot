## 2024-11-20 - [Accessibility in Python String Templates]
**Learning:** Python string templates (like those used in FastAPI `HTMLResponse`) are often missed by standard UI accessibility linters. This means accessibility bindings, such as `<label for="...">` and `<input id="...">`, can easily be overlooked.
**Action:** Always manually check and verify accessibility bindings when writing or modifying HTML strings directly within Python backends.
