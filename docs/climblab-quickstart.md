# CLIMBLAB quickstart

This guide is for CLIMBLAB members joining the lab's internal `nro`
installation. Outside users should follow the
[new-installation quickstart](quickstart.md). The shared software, paths,
registry, definitions, and Slurm worker pool are already configured.

## Connect your account

On a lab server, run:

```bash
cd /juice6/u/nlp/climblab/nro
./install --default
```

This creates `~/.local/bin/nro` and selects the lab installation as your
default. It does not create another Python environment or change shared
settings. Do not use `--maintain`; maintainers use it during coordinated
updates.

For Bash, add this line to `~/.bashrc` if needed:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Load and check the setup:

```bash
source ~/.bashrc
hash -r
type -a nro
nro doctor
nro paths show
```

The first `nro` found should be `~/.local/bin/nro`. No Conda or virtual
environment activation is required. If another command takes precedence,
deactivate that environment or remove the shell alias. Use
`~/.local/bin/nro` directly while correcting `PATH`.

## Inspect and request work

Always begin with explicit project and participant selectors. Project uses
uppercase `-P`; participant uses lowercase `-p`:

```bash
nro status -P nptl -p t20
nro run -P nptl -p t20 -m microparcellation
```

Replace the example values with the project and participant you need. The
planner adds upstream modules and reuses fresh results. Omitted selectors
usually mean all matches. A bare `nro run` can request work across the lab, and
a bare `nro purge` can delete every controlled derivative after confirmation.

Current defaults are workflow `main`, space `fsnative`, and 2 mm FWHM
smoothing. Multiple spaces and smoothing values request their cross-product:

```bash
nro run -P nptl -p t20 -m networks -s fsnative T1w -S 0 2
```

## Follow or stop a request

```bash
nro status -P nptl -p t20
nro log -P nptl -p t20 -m func -i
nro stop -P nptl -p t20
```

Status reports upstream errors that block downstream work. The `-i` log view
groups output by processing instance. Stop withdraws your matching demand
without deleting completed derivatives or another user's demand.

Read the [maintenance command guide](commands/maintenance.md) before using
purge. Contact the shared-installation maintainer if `nro doctor` fails or setup
is reported as incomplete. Onboarding does not require path edits, dependency
updates, registry repair, or changes to the shared checkout.

Continue with the [command reference](commands/index.md),
[module guides](modules/index.md), and [viewing guide](commands/viewing.md).
