
# reqgen

Generate a `requirements.txt` by scanning your Python code for imports. Also compare those requirements with your current environment (pip _or_ conda) to find missing packages and version differences.

## Install

```bash
pip install git+https://github.com/AbdulkadirUgas/reqgen.git

# or clone the project and install local dev
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

