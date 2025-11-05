import ast
import os
from pathlib import Path
import sys

# Find all Python files
base_path = Path("src/nifty_scalper_bot")
py_files = []
for folder in ["core", "execution", "infra", "data"]:
    folder_path = base_path / folder
    if folder_path.exists():
        py_files.extend(folder_path.glob("*.py"))

print(f"Found {len(py_files)} Python files:")
for f in py_files:
    print(f"  {f}")

# Try to parse each file and identify syntax errors
print("\n=== Analyzing files ===")
for py_file in py_files:
    print(f"\nChecking {py_file}...")
    try:
        with open(py_file, "r") as f:
            code = f.read()
            ast.parse(code)
        print(f"  ✓ Syntax OK")
    except SyntaxError as e:
        print(f"  ✗ SYNTAX ERROR: {e}")
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
