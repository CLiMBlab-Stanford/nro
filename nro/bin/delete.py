"""Delete one model, config, or workflow definition without removing derivatives."""

from nro.configuration.authoring import main as author


def main(argv: list[str] | None = None, *, prog: str = "nro.bin.delete") -> None:
    """Confirm removal of a definition and retain a temporary recovery copy."""
    author("delete", argv, prog=prog)


if __name__ == "__main__":
    main()
