# File Header Standard — legacy critical-file labels

> [MODULE_DOCUMENTATION.md](MODULE_DOCUMENTATION.md) is the canonical policy.
> This page retains strict three-label formatting for the subset already
> checked by `tests/architecture/test_file_header_standard.py`; other
> production Python modules may use equivalent descriptive docstrings.

Every changed production or deployment file needs a concise explanatory header under the canonical policy. The strict labels below remain required for the explicitly tested legacy critical-file subset.

## Required content

The header must contain these three labels:

- **File purpose:** why the file exists.
- **Key responsibilities:** the behaviour or configuration owned by the file.
- **Operational constraints:** safety boundaries, non-responsibilities, or invariants that must not be broken.

## Formats

### Python

Use the module docstring at the first line:

```python
"""File purpose:
    Brief purpose.

Key responsibilities:
    - Responsibility one.
    - Responsibility two.

Operational constraints:
    - Important invariant or ownership boundary.
"""
```

### Dockerfile, TOML, YAML and shell files

Use comments at the beginning of the file:

```text
# File purpose: Brief purpose.
# Key responsibilities: Main behaviour owned by this file.
# Operational constraints: Important invariant or safety boundary.
```

## Editing rule

When a listed canonical BO or deployment file is edited, its header must be reviewed and updated if ownership or behaviour changed. CI validates the required labels so later edits cannot remove the file-level explanation silently.
