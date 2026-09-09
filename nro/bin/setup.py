"""Set up an installation or connect to its shared environment."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from nro.configuration.site import CHECKOUT, installation_record, settings, site_file
from nro.engine.bootstrap import cancel_setup
from nro.engine.dependencies import (
    LICENSE_HELP,
    QUNEX_TERMS,
    check_installation,
    install_images,
    install_oslom,
    install_runtime,
    install_templates,
    install_workbench,
)
from nro.engine.site_setup import edit_settings, save_settings


def _main(argv=None, *, prog="nro setup"):
    values = list(sys.argv[1:] if argv is None else argv)
    if "--resources-only" not in values:
        result = subprocess.run(
            [sys.executable, str(CHECKOUT / "install"), *values],
            env={**os.environ, "NRO_SETUP_CHILD": "1"},
        )
        raise SystemExit(result.returncode)
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--resources-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--maintain", action="store_true")
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--without-oslom", action="store_true")
    parser.add_argument("--accept-qunex-license", action="store_true")
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args(values)
    if installation_record().get("mode") == "branch":
        if args.maintain:
            parser.error("Branch installations cannot maintain shared resources")
        results = check_installation(
            deep=False, with_oslom=not args.without_oslom, slurm=not args.local
        )
        for result in results:
            print(f"{'OK' if result['ok'] else 'MISSING'} {result['name']}: {result['detail']}")
        if any(not result["ok"] and result["required"] for result in results):
            parser.exit(
                1, "Shared resources are unavailable; ask the site maintainer to check them.\n"
            )
        return
    if installation_record().get("mode") == "shared" and not args.maintain:
        parser.error("Shared resource maintenance requires --maintain")
    try:
        from nro.engine.bootstrap import check_workers

        check_workers(site_file())
        path = site_file()
        if not path.exists():
            if args.non_interactive:
                save_settings(path, {})
            else:
                edit_settings(maintain=args.maintain)
                if not path.exists():
                    raise RuntimeError("Path setup was cancelled")
        site, _ = settings()
        from nro.configuration.definitions import ensure_store

        definitions = ensure_store(Path(site["definitions"]))
        print(f"Definitions: {definitions}", flush=True)
        print(f"Using {path}\nQuNex terms: {QUNEX_TERMS}", flush=True)
        if not Path(site["license"]).is_file():
            raise RuntimeError(LICENSE_HELP)
        if (
            args.non_interactive
            and not args.offline
            and not Path(site["qunex"]).is_file()
            and not args.accept_qunex_license
        ):
            raise RuntimeError(
                "Review the QuNex terms and pass --accept-qunex-license for unattended acquisition"
            )
        if not args.non_interactive and not args.offline:
            print(
                "Missing containers, Workbench, and templates will be downloaded to the configured locations."
            )
            print(f"Image destination: {site['images']}")
            for key in ("images", "templates", "workbench"):
                parent = Path(site[key])
                while not parent.exists():
                    parent = parent.parent
                print(f"{key}: {shutil.disk_usage(parent).free / 2**30:.1f} GiB free")
            if input("Proceed under the linked software terms? [y/N]: ").strip().lower() not in {
                "y",
                "yes",
            }:
                raise RuntimeError("Resource setup cancelled")
        install_runtime(offline=args.offline)
        install_images(offline=args.offline)
        install_workbench(offline=args.offline)
        install_templates(offline=args.offline)
        if not args.without_oslom:
            install_oslom(offline=args.offline)
        results = check_installation(
            deep=True, with_oslom=not args.without_oslom, slurm=not args.local
        )
        for result in results:
            print(f"{'OK' if result['ok'] else 'MISSING'} {result['name']}: {result['detail']}")
        if any(not r["ok"] and r["required"] for r in results):
            raise RuntimeError("Required checks failed; correct the settings and rerun ./install")
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Setup incomplete: {error}\n")


def main(argv=None, *, prog="nro setup"):
    """Run setup or shared onboarding, reporting cancellation once with status 130.

    argv excludes the executable name; None reads the process arguments.
    prog controls help/error labels. Invalid arguments raise SystemExit.
    """
    try:
        _main(argv, prog=prog)
    except (KeyboardInterrupt, EOFError):
        cancel_setup()
    except SystemExit as error:
        if error.code in {-2, 130}:
            cancel_setup()
        raise


if __name__ == "__main__":
    main()
