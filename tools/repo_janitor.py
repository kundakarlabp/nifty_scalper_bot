#!/usr/bin/env python3
from __future__ import annotations
import hashlib, os, sys, json, re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set

ROOT = Path(os.getenv("JANITOR_ROOT", "."))

IGNORE_DIRS = {".git", ".venv", "venv", "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache", "archive", "dist", "build"}
PY_EXT = {".py"}
BIG_FILE_MB = float(os.getenv("JANITOR_BIG_MB", "5"))

ENTRY_POINTS = [
    "src/main.py",
    "src/strategies/runner.py",
    "src/core/orchestrator.py",
]

def walk_files() -> List[Path]:
    files=[]
    for p in ROOT.rglob("*"):
        if p.is_dir():
            if p.name in IGNORE_DIRS: 
                # skip subtree
                parts = p.parts
                continue
            continue
        # skip binary-ish
        if any(part in IGNORE_DIRS for part in p.parts): 
            continue
        files.append(p)
    return files

def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256(); h.update(b); return h.hexdigest()

def file_hash(p: Path) -> str:
    try:
        return sha256_bytes(p.read_bytes())
    except Exception:
        return ""

def normalize_py(p: Path) -> str:
    """Rough near-duplicate: strip comments/blank lines, collapse whitespace."""
    try:
        s = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    # remove block & line comments (# …) naïvely
    s = re.sub(r"(?m)^\s*#.*$", "", s)
    s = re.sub(r'""".*?"""', "", s, flags=re.S)
    s = re.sub(r"'''.*?'''", "", s, flags=re.S)
    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return sha256_bytes(s.encode("utf-8"))

def detect_duplicates(files: List[Path]) -> Dict[str, List[str]]:
    groups = defaultdict(list)
    for p in files:
        h = file_hash(p)
        if h:
            groups[h].append(str(p))
    return {h: v for h, v in groups.items() if len(v) > 1}

def detect_near_duplicates_py(files: List[Path]) -> Dict[str, List[str]]:
    groups = defaultdict(list)
    for p in files:
        if p.suffix in PY_EXT:
            h = normalize_py(p)
            if h:
                groups[h].append(str(p))
    return {h: v for h, v in groups.items() if len(v) > 1}

def big_artifacts(files: List[Path]) -> List[Tuple[str, float]]:
    out=[]
    for p in files:
        try:
            mb = p.stat().st_size / (1024*1024)
            if mb >= BIG_FILE_MB:
                out.append((str(p), round(mb, 2)))
        except Exception:
            pass
    return sorted(out, key=lambda x: -x[1])

def import_graph(files: List[Path]) -> Tuple[Dict[str, Set[str]], Dict[str, str]]:
    """Very rough graph: module -> imported modules (by file stem)."""
    mod_by_file={}
    for p in files:
        if p.suffix in PY_EXT:
            # module key as path sans .py
            mod_by_file[str(p)] = str(p.with_suffix(""))
    deps=defaultdict(set)
    pat = re.compile(r"^\s*(?:from\s+([\w\.]+)\s+import|import\s+([\w\.]+))", re.M)
    for p in files:
        if p.suffix not in PY_EXT: continue
        try: s = p.read_text(encoding="utf-8", errors="ignore")
        except Exception: continue
        for m in pat.finditer(s):
            name = (m.group(1) or m.group(2) or "").split(".")[0]
            if name: deps[str(p)].add(name)
    return deps, mod_by_file

def find_orphans(files: List[Path]) -> List[str]:
    """Heuristic: python files not reachable from entry points by stem imports."""
    py_files = [str(p) for p in files if p.suffix in PY_EXT]
    deps, mod_by_file = import_graph(files)
    # build map stem -> files
    by_stem=defaultdict(list)
    for f in py_files:
        stem = Path(f).stem
        by_stem[stem].append(f)

    # start set: entry points + tests
    roots = set(f for f in py_files if any(f.endswith(ep) for ep in ENTRY_POINTS) or f.startswith("tests/"))
    reachable=set(roots)
    # BFS by stem references
    queue=list(roots)
    while queue:
        f = queue.pop()
        for dep in deps.get(f, set()):
            for tgt in by_stem.get(dep, []):
                if tgt not in reachable:
                    reachable.add(tgt); queue.append(tgt)
    # orphans = py_files - reachable, but keep __init__.py
    orphans=[f for f in py_files if f not in reachable and not f.endswith("__init__.py")]
    return sorted(orphans)

def main():
    files = walk_files()
    dups = detect_duplicates(files)
    near = detect_near_duplicates_py(files)
    bigs = big_artifacts(files)
    orph = find_orphans(files)

    report = {
        "duplicates_exact": list(dups.values()),
        "duplicates_near_py": list(near.values()),
        "big_files_over_mb": bigs,
        "orphans_py": orph,
    }
    Path("JANITOR_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
