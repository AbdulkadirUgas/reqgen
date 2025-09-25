# Project: reqgen
# A CLI tool to generate requirements.txt by scanning Python files and to compare
# them with your current environment (pip/conda). Installable from GitHub.
#
# ├── pyproject.toml
# └── src/
#     └── reqgen/
#         ├── __init__.py
#         ├── cli.py
#         ├── scan.py
#         └── mappings.py
#
# Paste these files in a repo with this structure. Then:
#   pip install -e .
#   reqgen --help

# =========================
# File: pyproject.toml
# =========================

pyproject_toml = r"""
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "reqgen"
version = "0.1.0"
description = "Generate requirements.txt from your codebase and compare with current environment"
readme = "README.md"
authors = [{name = "Your Name", email = "you@example.com"}]
requires-python = ">=3.9"
license = {text = "MIT"}
keywords = ["requirements.txt", "dependency", "scanner", "pip", "conda"]
classifiers = [
  "Programming Language :: Python :: 3",
  "License :: OSI Approved :: MIT License",
  "Environment :: Console",
  "Topic :: Software Development :: Build Tools",
]

[project.scripts]
reqgen = "reqgen.cli:main"

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
include = ["reqgen*"]
"""

# =========================
# File: src/reqgen/__init__.py
# =========================

init_py = r"""
__all__ = ["generate_requirements", "compare_environment"]
from .scan import generate_requirements, compare_environment
"""

# =========================
# File: src/reqgen/mappings.py
# =========================

mappings_py = r"""
# Heuristic map from import name -> PyPI distribution name for common cases
# Extend as needed.
IMPORT_TO_DIST = {
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "bs4": "beautifulsoup4",
    "yaml": "PyYAML",
    "Crypto": "pycryptodome",
    "yaml": "PyYAML",
    "lxml": "lxml",
    "ujson": "ujson",
    "orjson": "orjson",
    "tomli": "tomli",
    "tomllib": "tomllib",  # stdlib in 3.11+, keep for older pins
    "IPython": "ipython",
    "jupyter": "jupyter",
    "notebook": "notebook",
    "tensorflow": "tensorflow",
    "tf_keras": "tf-keras",
    "torch": "torch",
    "torchvision": "torchvision",
    "torchaudio": "torchaudio",
    "jax": "jax",
    "jaxlib": "jaxlib",
    "cupy": "cupy-cuda11x",
    "pymongo": "pymongo",
    "psycopg2": "psycopg2-binary",
    "pymysql": "pymysql",
    "mysqlclient": "mysqlclient",
    "sqlalchemy": "SQLAlchemy",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "starlette": "starlette",
    "django": "Django",
    "flask": "Flask",
    "jinja2": "Jinja2",
    "pydantic": "pydantic",
    "requests": "requests",
    "httpx": "httpx",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "tqdm": "tqdm",
    "rich": "rich",
    "typer": "typer",
}
"""

# =========================
# File: src/reqgen/scan.py
# =========================

scan_py = r"""
from __future__ import annotations
import ast
import os
import sys
from dataclasses import dataclass
from typing import Iterable, Optional, Set, Dict, Tuple

try:
    from importlib import metadata as importlib_metadata  # py3.8+
except ImportError:  # pragma: no cover
    import importlib_metadata  # type: ignore

try:
    STDLIB = set(sys.stdlib_module_names)  # py3.10+
except Exception:
    # Fallback minimal set; users on older Pythons can extend if needed
    STDLIB = {
        "sys","os","re","math","json","pathlib","subprocess","itertools","functools",
        "collections","datetime","typing","argparse","logging","uuid","enum","dataclasses",
        "ast","asyncio","hashlib","heapq","inspect","io","glob","shutil","tempfile","threading",
    }

from .mappings import IMPORT_TO_DIST

@dataclass(frozen=True)
class ScanResult:
    imports: Set[str]            # import module names (top-level)
    dists: Dict[str, Optional[str]]  # module -> dist (None if unknown)

@dataclass
class Requirements:
    pins: Dict[str, Optional[str]]  # dist -> version or None


def _iter_py_files(root: str) -> Iterable[str]:
    for dirpath, dirnames, filenames in os.walk(root):
        # skip typical virtual envs and hidden dirs
        if any(skip in dirpath for skip in (os.sep+".git", os.sep+".venv", os.sep+"venv", os.sep+"__pycache__")):
            continue
        for f in filenames:
            if f.endswith(".py"):
                yield os.path.join(dirpath, f)


def _parse_imports(path: str) -> Set[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        try:
            tree = ast.parse(fh.read(), filename=path)
        except SyntaxError:
            return set()
    names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top and top not in (".", ".."):
                    names.add(top)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                # relative import -> probably local package
                continue
            if node.module:
                top = node.module.split(".")[0]
                names.add(top)
    return names


def scan_imports(root: str, single_file: Optional[str] = None) -> ScanResult:
    imports: Set[str] = set()
    files = [single_file] if single_file else list(_iter_py_files(root))
    for path in files:
        if not path:  # safety
            continue
        if os.path.isfile(path) and path.endswith(".py"):
            imports |= _parse_imports(path)
    # remove stdlib and obvious locals
    filtered = {m for m in imports if m not in STDLIB}

    # map module -> dist (installed or heuristic)
    try:
        pkg_map = importlib_metadata.packages_distributions()  # module -> [dist]
    except Exception:
        pkg_map = {}

    dists: Dict[str, Optional[str]] = {}
    for mod in sorted(filtered):
        dist: Optional[str] = None
        if mod in pkg_map:
            dist = sorted(pkg_map[mod])[0]
        elif mod in IMPORT_TO_DIST:
            dist = IMPORT_TO_DIST[mod]
        else:
            # heuristic: assume module == dist name (often true)
            dist = mod
        dists[mod] = dist
    return ScanResult(imports=filtered, dists=dists)


def generate_requirements(root: str, single_file: Optional[str] = None, pin: str = "installed") -> Requirements:
    scan = scan_imports(root, single_file)
    pins: Dict[str, Optional[str]] = {}
    for mod, dist in scan.dists.items():
        if not dist:
            continue
        version: Optional[str] = None
        if pin == "installed":
            try:
                version = importlib_metadata.version(dist)
            except importlib_metadata.PackageNotFoundError:
                version = None
        pins[dist] = version
    return Requirements(pins=pins)


def compare_environment(root: str, single_file: Optional[str] = None, pin: str = "installed") -> Tuple[Dict[str, str], Dict[str, Tuple[str, str]]]:
    reqs = generate_requirements(root, single_file=single_file, pin=pin)

    env_versions: Dict[str, str] = {}
    for d in importlib_metadata.distributions():
        try:
            name = d.metadata["Name"] or d.metadata["Summary"]  # type: ignore
        except Exception:
            name = d.metadata.get("Name", d.metadata.get("Summary", ""))  # type: ignore
        name = (name or d.metadata.get("Name") or d.metadata.get("Summary") or d.metadata.get("Summary", "")).strip()  # type: ignore
        if not name:
            name = d.metadata.get("Name", "")  # type: ignore
        env_versions[name or d.metadata.get("Name", "")] = d.version  # type: ignore
        # Normalize key to canonical case-insensitive form
        env_versions[d.metadata.get("Name", d.metadata.get("Summary", d.metadata.get("Name", ""))).lower()] = d.version  # type: ignore

    # Normalize requirement keys to lowercase for comparison
    normalized_reqs: Dict[str, Optional[str]] = {k.lower(): v for k, v in reqs.pins.items()}

    missing: Dict[str, str] = {}
    mismatches: Dict[str, Tuple[str, str]] = {}

    for dist_lower, req_ver in normalized_reqs.items():
        env_ver = env_versions.get(dist_lower)
        if env_ver is None:
            missing[dist_lower] = req_ver or "*"
        else:
            if req_ver and req_ver != env_ver:
                mismatches[dist_lower] = (req_ver, env_ver)

    return missing, mismatches
"""

# =========================
# File: src/reqgen/cli.py
# =========================

cli_py = r"""
from __future__ import annotations
import argparse
import os
from typing import Optional

from .scan import generate_requirements, compare_environment


def _write_requirements(path: str, pins: dict[str, str|None]) -> None:
    lines = []
    for dist, ver in sorted(pins.items(), key=lambda kv: kv[0].lower()):
        if ver:
            lines.append(f"{dist}=={ver}")
        else:
            lines.append(dist)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def cmd_scan(args: argparse.Namespace) -> int:
    pin = args.pin
    reqs = generate_requirements(args.path, single_file=args.file, pin=pin)
    if args.output:
        _write_requirements(args.output, reqs.pins)
        print(f"Wrote {len(reqs.pins)} requirements to {args.output}")
    else:
        for dist, ver in sorted(reqs.pins.items()):
            print(f"{dist}{'=='+ver if ver else ''}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    missing, mismatches = compare_environment(args.path, single_file=args.file, pin=args.pin)
    if not missing and not mismatches:
        print("✔ Environment satisfies the project's imports.")
        return 0
    if missing:
        print("Missing packages:")
        for dist, ver in sorted(missing.items()):
            print(f"  - {dist}{'=='+ver if ver and ver!='*' else ''}")
    if mismatches:
        print("Version differences (required vs env):")
        for dist, (need, have) in sorted(mismatches.items()):
            print(f"  - {dist}: {need} (required) != {have} (installed)")
    return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="reqgen", description="Generate requirements.txt from imports and compare with your environment")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("scan", help="Scan project or single file and print/write requirements")
    ps.add_argument("path", nargs="?", default=os.getcwd(), help="Project root (default: cwd)")
    ps.add_argument("--file", "-f", help="Scan only this single .py file")
    ps.add_argument("--output", "-o", help="Write to this requirements.txt path")
    ps.add_argument("--pin", choices=["none", "installed"], default="installed", help="Version pinning strategy")
    ps.set_defaults(func=cmd_scan)

    pc = sub.add_parser("compare", help="Compare project requirements with current environment")
    pc.add_argument("path", nargs="?", default=os.getcwd(), help="Project root (default: cwd)")
    pc.add_argument("--file", "-f", help="Compare for a single .py file")
    pc.add_argument("--pin", choices=["none", "installed"], default="installed", help="How to derive required versions")
    pc.set_defaults(func=cmd_compare)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)

if __name__ == "__main__":
    raise SystemExit(main())
"""

# =========================
# File: README.md (optional but recommended)
# =========================

readme_md = r"""
# reqgen

Generate a `requirements.txt` by scanning your Python code for imports. Also compare those requirements with your current environment (pip _or_ conda) to find missing packages and version differences.

## Install

```bash
pip install git+https://github.com/<you>/reqgen.git
# or local dev
pip install -e .
```

## Usage

Scan whole project (current directory) and write `requirements.txt` with versions pinned to what's installed:

```bash
reqgen scan -o requirements.txt
```

Scan only one file:

```bash
reqgen scan --file path/to/script.py -o requirements.txt
```

Print requirements without writing a file and without versions:

```bash
reqgen scan --pin none
```

Compare with your current environment:

```bash
reqgen compare
```

This prints missing packages and version mismatches.

## Notes
- Works in pip or conda: versions are discovered via `importlib.metadata`.
- Heuristics map common import names to PyPI packages (e.g., `cv2 -> opencv-python`, `PIL -> Pillow`). Extend `reqgen/mappings.py` for more.
- Relative imports are treated as local modules and ignored.

"""

# --- Below: write the files into a working directory if you run this cell ---
if __name__ == "__main__":
    import os, pathlib
    root = pathlib.Path("./reqgen_repo").resolve()
    (root / "src" / "reqgen").mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(pyproject_toml, encoding="utf-8")
    (root / "README.md").write_text(readme_md, encoding="utf-8")
    (root / "src" / "reqgen" / "__init__.py").write_text(init_py, encoding="utf-8")
    (root / "src" / "reqgen" / "mappings.py").write_text(mappings_py, encoding="utf-8")
    (root / "src" / "reqgen" / "scan.py").write_text(scan_py, encoding="utf-8")
    (root / "src" / "reqgen" / "cli.py").write_text(cli_py, encoding="utf-8")
    print(f"Wrote skeleton to {root}")
