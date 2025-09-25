
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
