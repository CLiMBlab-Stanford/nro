# Development and documentation

This is an editable, early-development codebase. Public signatures are documented
but not version-stable. Coordinate shared changes and stop workers before
updating the installation. Do not develop experimental modules against the
production worker pool as if revisions were isolated.

## Tests

Install with `--dev` to include the locked test dependencies:

```bash
./install --maintain --dev
.nro-env/bin/python -m pytest -q
```

Installation tests isolate site settings and mock scheduler mutations. Scientific
unit tests use small synthetic inputs; they do not replace visual inspection
and deployment-specific container tests on real acquisitions.

## Documentation build

```bash
python3.12 -m venv /tmp/nro-docs
/tmp/nro-docs/bin/pip install -r docs/requirements.txt
/tmp/nro-docs/bin/sphinx-build -W --keep-going -b html docs docs/_build/html
```

Open `docs/_build/html/index.html`. Read the Docs uses `.readthedocs.yaml` with
the same requirements and treats build warnings as failures. Connecting the
repository to a Read the Docs project is a separate hosting action; this
configuration does not create a hosted site.

Markdown pages use MyST. AutoAPI reads Python source without importing the
scientific package. Add docstrings for public classes and methods where the
code is defined; API pages are generated and should not be edited directly.
Configuration examples are included from their source YAML. When changing a
step, update its method page, configuration meaning, artifact contract, and tests.

Follow `WRITING_POLICY.md` for prose, `CONTRIBUTING.md` for attribution and
publication, and `AGENTS.md` for agent instructions. No documentation build
should modify a live registry or download scientific images.
