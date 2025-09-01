import os


def env_flag(name: str, default: bool = True) -> bool:
    """Parse an environment variable into a boolean flag.

    Accepts typical truthy/falsey string values like "1", "0", "true", "false",
    "yes", "no", "on", and "off" (case-insensitive). Returns ``default`` when the
    variable is missing or cannot be interpreted.
    """
    val = os.environ.get(name)
    if val is None:
        return default
    norm = str(val).strip().lower()
    if norm in {"1", "true", "yes", "on"}:
        return True
    if norm in {"0", "false", "no", "off"}:
        return False
    return default
