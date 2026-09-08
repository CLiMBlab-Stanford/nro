"""Check dependencies and site access without modifying installation state."""

import argparse
import json
from nro.engine.dependencies import check_installation


def main(argv=None, *, prog="nro doctor"):
    """Report dependency checks and exit nonzero when a required check fails.

    argv excludes the executable name; None reads the process arguments.
    prog controls help/error labels. Invalid arguments raise SystemExit.
    """
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--deep", action="store_true", help="Also execute probes inside containers")
    parser.add_argument("--without-oslom", action="store_true", help="Treat OSLOM as optional")
    parser.add_argument("--local", action="store_true", help="Treat Slurm as optional")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    results = check_installation(deep=args.deep, with_oslom=not args.without_oslom, slurm=not args.local)
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for result in results:
            status = "OK" if result["ok"] else "FAIL" if result["required"] else "OPTIONAL"
            print(f"{status:8} {result['name']}: {result['detail']}")
    if any(not row["ok"] and row["required"] for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
