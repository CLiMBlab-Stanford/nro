# Develop on a branch

This tutorial walks a developer through the life of an nro branch: creating and
registering it, testing on it, merging it into its parent, and releasing `dev`
into `main`.

nro isolates development code and outputs by Git branch. Each registered branch
has one scientific registry that all of its checkouts share. A development
branch inherits compatible artifacts from its registered ancestors and writes
new artifacts beneath the site's development directory.

The steps assume that a tagged `main` release is installed as the shared
scheduler. They use `dev` as the integration branch and `feature/example` as a
feature branch. Replace the paths, names, versions, and pull-request references
with your own.

## Create and register a branch

Use a separate checkout for each branch. An installed checkout is bound to the
branch it had at installation, and nro rejects the installation if you switch
that checkout to another branch. A Git worktree gives you a separate checkout
without another full clone:

```bash
cd /path/to/nro-dev
git fetch origin
git pull --ff-only origin dev
git worktree add -b feature/example /path/to/nro-feature-example dev
cd /path/to/nro-feature-example
./install --mode branch --dev
nro branch show
```

`git worktree add` creates the Git branch and checks it out. `./install` then
creates the checkout's editable environment, connects it to the shared site,
and registers the branch with parent `dev`. If the branch is already
registered, the installer attaches the new checkout to the existing scientific
registry instead. The `--dev` option installs test and lint dependencies. It
does not refer to the Git branch named `dev`.

Most users need only the installer. You can also register a branch explicitly
by running `nro branch register` through an nro installation that already
works:

```bash
cd /tmp
nro branch register --checkout /path/to/nro-feature-example --parent dev
```

Explicit registration helps when an environment is already prepared or when you
need to record a parent other than `dev` before setup. A later installer run
recognizes the registration and attaches the checkout. To add another checkout
for a branch that is already registered, use `nro branch attach --checkout PATH`.

Branch names are reserved across the whole shared nro installation, and retiring
a branch does not release its name. Choose a unique, descriptive Git branch
name.

### Base a branch on another development branch

The Git base and the nro parent should describe the same relationship. To base
a branch on another development branch, register the parent explicitly before
installing:

```bash
cd /path/to/nro-parent
git worktree add -b feature/child /path/to/nro-child feature/parent

cd /tmp
nro branch register --checkout /path/to/nro-child --parent feature/parent

cd /path/to/nro-child
./install --mode branch --dev
nro branch show
```

If the installer registered a new branch under its default parent, `dev`, change
the relationship before requesting work:

```bash
nro branch reparent feature/child --parent feature/parent
```

Run `reparent` from an attached checkout of the branch you are changing. It
updates artifact inheritance and reconciles affected requests. It does not
change Git history, move outputs, or merge code.

## Develop and test

Ordinary nro commands select the branch from the current working directory:

```bash
nro doctor
nro status -P nptl -p t20
nro run -P nptl -p t20 -m networks
```

The feature branch runs its captured code and environment, reads compatible
ancestor artifacts, and writes new outputs to its branch directory. The central
scheduler still applies the lab-wide concurrency limit. Run `nro branch show` to
confirm the selected branch, parent, registry, and definitions store.

By default, a development branch reads the shared definitions store without
write access. If your feature changes configs, workflows, models, or events,
select a branch-specific definitions checkout:

```bash
nro branch definitions --definitions /path/to/feature-definitions
```

Return to the shared definitions with `nro branch definitions --shared`. See
[branch registration](commands/branches.md#development-definitions) for the
ownership and validation rules.

## Merge one development branch into another

The destination should be an ancestor of the source in both Git and nro. If the
destination changed during development, reparent the source branch and repeat
its relevant tests before you open the pull request.

Commit and push the source branch through the normal Git workflow:

```bash
cd /path/to/nro-feature-example
git status
git add PATHS_TO_COMMIT
git commit
git push -u origin feature/example
gh pr create --base dev --head feature/example
```

Set `--base` to the registered development parent. Follow the
[contribution policy](https://github.com/CLiMBlab-Stanford/nro/blob/dev/CONTRIBUTING.md)
for authorship, review, and AI assistance metadata. After approval, merge the
pull request with a merge commit through GitHub, then update the parent's
dedicated checkout:

```bash
gh pr merge --merge feature/example
cd /path/to/nro-dev
git fetch origin
git merge --ff-only origin/dev
```

The Git merge makes the code available to the parent but does not copy any
development artifacts. If the source branch produced expensive artifacts that
are still scientifically equivalent under the merged code, preview and then
promote them from the accepting checkout:

```bash
nro promote --from feature/example --pr CLiMBlab-Stanford/nro#123 \
  --attest-merged -P nptl -p t20 -m networks --dry-run
nro promote --from feature/example --pr CLiMBlab-Stanford/nro#123 \
  --attest-merged -P nptl -p t20 -m networks
```

Promotion is optional. Without it, the parent reuses its existing fresh outputs
and computes missing or stale work on demand. `nro promote` verifies current
contracts and copies eligible artifacts. It does not merge Git branches or
delete source files.

Before you retire the merged branch, reparent any active child branches. Run
this from each child's attached checkout:

```bash
cd /path/to/nro-child
nro branch reparent feature/child --parent dev
```

Then retire the merged branch from its own checkout:

```bash
cd /path/to/nro-feature-example
nro branch retire feature/example
```

Retirement cancels the branch's demand and prevents future checkouts. It keeps
the branch's registry and outputs, so purge any branch-owned artifacts you do
not want to keep before retiring. Removing the Git worktree and remote branch
are separate steps. Do them only after the working tree is clean and the merge
is confirmed.

## Merge `dev` into `main`

`main` contains releases only. Every pull request into `main` must advance the
semantic version in `pyproject.toml` by at least one patch.

Prepare the release on `dev`. Update its checkout:

```bash
cd /path/to/nro-dev
git fetch origin
git pull --ff-only origin dev
```

Edit the version in `pyproject.toml` and the applicable release documentation,
and run focused checks for the change. Then commit, push `dev`, and open the
pull request. GitHub runs the complete release suite before the pull request can
merge:

```bash
git add pyproject.toml PATHS_TO_RELEASE_NOTES
git commit
git push origin dev
gh pr create --base main --head dev
```

After approval, merge the pull request with a merge commit. This preserves
ancestry and lets `dev` fast-forward to the released commit afterward. Update
the dedicated shared `main` checkout, create an annotated tag, and push it:

```bash
gh pr merge --merge dev
cd /path/to/nro-main
git fetch origin
git merge --ff-only origin/main
git tag -a vMAJOR.MINOR.PATCH -m "nro MAJOR.MINOR.PATCH"
git push origin vMAJOR.MINOR.PATCH
```

Replace `MAJOR.MINOR.PATCH` with the version in `pyproject.toml`. Pushing a
valid version tag starts the GitHub Release workflow. Confirm that the workflow
created the corresponding GitHub Release, then edit the release so that it
begins with a concise, human-written summary of the important changes, as the
[contribution policy](https://github.com/CLiMBlab-Stanford/nro/blob/dev/CONTRIBUTING.md#releases-and-compatibility)
requires. Do not move a published tag.

Install and activate the new release from the shared `main` checkout:

```bash
./install --maintain
nro release
nro doctor
```

The installer checks that `main` is clean, tagged, and present on `origin/main`.
It records the release and activates that checkout for future scheduler work. If
work is active, it offers to let current attempts finish or to stop them
immediately, and either way it keeps their demand. No separate
branch-registration or release-approval command is needed.

After activation, you may promote equivalent `dev` artifacts using the accepted
pull-request reference:

```bash
nro promote --from dev --pr CLiMBlab-Stanford/nro#124 --attest-merged \
  -P nptl -p t20 -m networks --dry-run
```

Finally, fast-forward the dedicated `dev` checkout to the release commit and
push it, so that new feature work starts from the current release:

```bash
cd /path/to/nro-dev
git fetch origin
git merge --ff-only origin/main
git push origin dev
```

Do not retire `dev`. It is the permanent integration branch beneath `main` in
nro's fixed branch topology.

## Command responsibilities

Git and nro track different state. This table shows which tool changes what:

| Operation | Command | Effect |
| --- | --- | --- |
| Create, commit, review, or merge code | `git` and the Git hosting service | Changes repository history. |
| Prepare and register a development checkout | `./install --mode branch` | Creates the environment and binds the checkout to the shared branch registry. |
| Register or attach without installation | `nro branch register` or `nro branch attach` | Changes checkout authorization and scientific-registry routing. |
| Change inheritance | `nro branch reparent` | Changes the nro parent tree and reconciles affected work. |
| Adopt equivalent artifacts after a merge | `nro promote` | Copies verified outputs into the accepting ancestor's ownership. |
| Close a development branch | `nro branch retire` | Cancels its demand and prevents later attachment while retaining records. |
| Activate a tagged `main` release | `./install --maintain` | Records the release and updates the shared scheduler implementation. |

Git operations never update nro's branch catalog. nro branch operations never
create, switch, merge, or delete Git branches.
