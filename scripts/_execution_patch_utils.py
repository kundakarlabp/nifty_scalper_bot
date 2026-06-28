from __future__ import annotations

import ast
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, content: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def replace_once(path: str, old: str, new: str) -> None:
    text = read(path)
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one anchor, found {count}: {old[:100]!r}")
    write(path, text.replace(old, new, 1))


def method_text(source: str) -> str:
    return textwrap.indent(textwrap.dedent(source).strip("\n") + "\n", "    ")


def replace_method(path: str, class_name: str, method_name: str, source: str) -> None:
    text = read(path)
    tree = ast.parse(text)
    cls = next(
        (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name),
        None,
    )
    if cls is None:
        raise RuntimeError(f"{path}: class {class_name} not found")
    node = next(
        (
            item
            for item in cls.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name == method_name
        ),
        None,
    )
    if node is None or node.end_lineno is None:
        raise RuntimeError(f"{path}: method {class_name}.{method_name} not found")
    lines = text.splitlines(keepends=True)
    lines[node.lineno - 1 : node.end_lineno] = [method_text(source)]
    write(path, "".join(lines))


def assert_parses(*paths: str) -> None:
    for path in paths:
        ast.parse(read(path), filename=path)
