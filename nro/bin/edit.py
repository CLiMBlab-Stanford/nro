"""Edit a stored model, config, workflow, or markup through a reviewed copy."""

from nro.configuration.authoring import main as author


def main(argv: list[str] | None = None, *, prog: str = "nro.bin.edit") -> None:
    """Validate and confirm changes to an existing definition."""
    author("edit", argv, prog=prog)


if __name__ == "__main__":
    main()
