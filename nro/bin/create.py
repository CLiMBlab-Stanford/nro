"""Create a model, config, or workflow; interactively edit existing definitions."""

from nro.configuration.authoring import main as author


def main(argv: list[str] | None = None, *, prog: str = "nro.bin.create") -> None:
    """Draft and validate a definition before confirmed publication."""
    author("create", argv, prog=prog)


if __name__ == "__main__":
    main()
