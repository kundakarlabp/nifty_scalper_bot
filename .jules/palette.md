## 2024-05-19 - Missing Label Associations in HTML Templates
**Learning:** In string-interpolated HTML templates (like those used in FastAPI dashboard views without a dedicated frontend framework), missing `for` and `id` attributes on form inputs can easily slip by since there are no automated frontend accessibility linters.
**Action:** Always ensure that manually constructed HTML strings in Python string templates map labels to their corresponding input elements using explicit `for` and `id` attributes to maintain screen reader support.
