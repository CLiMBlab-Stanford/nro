# CLIMBLAB quickstart

This guide is for CLIMBLAB members who want to use the lab's internal `nro`
installation. The processing software, data paths, registry, definitions, and
Slurm worker pool are already configured, so you only need to connect your
account. Outside users should follow the
[new-installation quickstart](quickstart.md).

## Connect your account

Log in to a lab server and run the installer from the shared checkout:

```bash
cd /juice6/u/nlp/climblab/nro
./install --default
```

This creates the launcher `~/.local/bin/nro` and selects the shared checkout as
your default installation. It does not create another Python environment,
download software, or change shared settings. Do not use `--maintain`.
Maintainers use it during coordinated updates of the shared installation.

If `~/.local/bin` is not already on your `PATH`, add this line to `~/.bashrc`:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Then load the change in your current shell:

```bash
source ~/.bashrc
hash -r
```

You do not need to activate a Conda or Python virtual environment. The launcher
selects the environment maintained with the shared checkout.

## Check the setup

Confirm which command your shell runs, then check the installation and the
configured lab paths:

```bash
type -a nro
nro doctor
nro site ls
```

The first path reported by `type -a nro` should be `~/.local/bin/nro`.
`nro doctor` should report `OK` for every required check. `nro site ls` should
show the shared CLIMBLAB data, work, definitions, TemplateFlow, and registry
locations. These settings live in the version-controlled
`nro-definitions/site/site.yml`. The checkout's site TOML file only points to
that repository.

If another `nro` appears first, an activated environment, shell alias, or shell
function is taking precedence. Deactivate the environment or remove the alias,
then run `hash -r`. You can use `~/.local/bin/nro` directly while you fix your
shell configuration.

If `nro doctor` fails or reports that setup is incomplete, contact the
maintainer of the shared installation. Onboarding never requires editing paths,
updating dependencies, repairing the registry, or changing the shared checkout.

## Inspect existing work

Most commands accept the same selectors. Project uses uppercase `-P` and
participant uses lowercase `-p`:

```bash
nro status -P nptl -p t20
```

Replace `nptl` and `t20` with the project and participant you need. Participant
labels may include or omit the `sub-` prefix. By default, `nro status` quickly
reports the registry's saved state without checking files. Run
`nro status --update` when you need it to check files and update that state.

Omitted selectors mean **all matching values**. A new `nro run` request has one
exception: it uses workflow `main` when `--workflow` is omitted, so it does not
request every defined workflow. Even so, a bare `nro run` can request `main`
work across the whole lab, and a bare `nro purge` can delete every derivative
under `nro` control after confirmation. With `nro run --resume`, an omitted
workflow again matches every workflow with resumable work. During onboarding,
always give a project and participant.

## Request processing

Request work by naming an endpoint module. The planner adds the upstream
modules it needs and reuses existing fresh results:

```bash
nro run -P nptl -p t20 -m microparcellation
```

You do not need to list upstream modules. For example,
`-m microparcellation` includes the required `anat`, `func`, and `clean` work.
Without `-m`, `run` requests every endpoint in the selected workflow.

The current defaults are workflow `main`, space `fsnative`, and 2 mm FWHM
smoothing. Select other spaces or smoothing values only when you need them.
Multiple values request their cross-product:

```bash
nro run -P nptl -p t20 -m networks -s fsnative T1w -S 0 2
```

Run `nro run --help` to see the task, model, run, workflow, and other
selectors.

## Follow or stop a request

Check status or read logs for the selected work:

```bash
nro status -P nptl -p t20
nro log -P nptl -p t20 -m func
```

Status shows blocked work and the upstream error that blocked it. The default
log view groups output by processing work item.

To withdraw your matching request without deleting completed derivatives, run:

```bash
nro stop -P nptl -p t20
```

Other users' requests for the same work stay active. Do not use `nro purge`
until you have read the [maintenance command guide](commands/maintenance.md).
Purge removes shared derivatives and logs, not just your request.

## Next steps

- [Command reference](commands/index.md)
- [Processing module guides](modules/index.md)
- [Viewing completed outputs](commands/viewing.md)
- [Task models](task-models.md)
