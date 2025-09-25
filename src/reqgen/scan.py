
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
    
    # Exclude modules that are likely local to the project
    local_files: Set[str] = set()
    local_dirs: Set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root):
        if any(skip in dirpath for skip in (os.sep + ".git", os.sep + ".venv", os.sep+"venv", os.sep+"__pycache__")):
            continue
        
        for d in dirnames:
            local_dirs.add(d)
        for f in filenames:
            if f.endswith(".py") and f != "__init__.py":
                local_files.add(f[:-3])
    
    filtered = {m for m in filtered if m not in local_files and m not in local_dirs}
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
