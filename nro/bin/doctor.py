"""Check dependencies and site access without modifying installation state."""

import argparse
import json
import sys

from nro.configuration import site
from nro.engine.dependencies import check_installation


def installation_details() -> list[dict]:
    """Describe the checkout and interpreter selected for this invocation."""
    record = site.installation_record()
    mode = record.get("mode", "unrecorded")
    branch = record.get("branch") or ("main" if mode == "shared" else None)
    values = {
        "checkout": str(site.CHECKOUT),
        "mode": mode,
        "branch": branch or "none",
        "environment": record.get("environment", str(sys.prefix)),
        "executable": sys.executable,
        "site settings": str(site.site_file()),
    }
    return [
        {
            "name": name,
            "ok": True,
            "required": True,
            "detail": value,
            "category": "installation",
        }
        for name, value in values.items()
    ]


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
    selected = installation_details()
    results = selected + check_installation(
        deep=args.deep,
        with_oslom=not args.without_oslom,
        slurm=not args.local,
        quick=not args.deep,
    )
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for result in results:
            if result.get("category") == "installation":
                status = "SELECTED"
            elif result["ok"]:
                status = "OK"
            else:
                status = "FAIL" if result["required"] else "OPTIONAL"
            print(f"{status:8} {result['name']}: {result['detail']}")
    if any(not row["ok"] and row["required"] for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
