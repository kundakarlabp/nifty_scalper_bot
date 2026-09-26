# Machine-readable runtime contracts

These schemas document existing repository contracts for coding agents and tests. They do **not** create a parallel runtime model.

Rules:

- Runtime owners remain authoritative.
- Schemas should describe stable cross-module fields, not every incidental implementation detail.
- Samples under `tests/fixtures/contracts/` must validate against these schemas.
- If a runtime owner changes a documented field, update the schema and sample in the same PR.
- Prefer backward-compatible additions over renaming/removing established fields.

The catalog is `contract_manifest.json`.
