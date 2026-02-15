"""Find likely missing await usages for known async APIs."""

from __future__ import annotations

from pathlib import Path

ASYNC_FUNCTIONS = ("get_positions", "fetch_history", "update_order_book")
SCAN_ROOTS = ("src",)
SKIP_PARTS = (".venv", "__pycache__", ".git", "tests")


def should_skip(path: Path) -> bool:
    """Args: path; Returns: bool; Raises: none."""

    return any(part in SKIP_PARTS for part in path.parts)


def main() -> int:
    """Args: none; Returns: exit code; Raises: none."""

    findings: list[str] = []
    for root in SCAN_ROOTS:
        for path in Path(root).rglob("*.py"):
            if should_skip(path):
                continue
            for line_no, line in enumerate(path.read_text().splitlines(), start=1):
                stripped = line.strip()
                if stripped.startswith("async def "):
                    continue
                for name in ASYNC_FUNCTIONS:
                    token = f"{name}("
                    if token in stripped and "await " not in stripped:
                        findings.append(
                            f"{path}:{line_no}: add await before {name} call"
                        )
    if not findings:
        print("No obvious missing await usages found.")
        return 0
    print("\n".join(findings))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
