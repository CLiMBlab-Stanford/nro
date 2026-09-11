# Develop on a branch

nro uses Git branches to isolate development code and outputs. Each registered
branch has one scientific registry shared by all of its checkouts. Development
branches inherit compatible artifacts from their registered ancestors and write
new artifacts beneath the site's development directory.

This tutorial assumes that a tagged `main` release is installed as the shared
scheduler. It uses `dev` as the integration branch and `feature/example` as a
feature branch. Replace paths, names, versions, and pull-request references with
the values for your work.

## Create and register a branch

Use a separate checkout for each branch. An installed checkout is bound to its
current branch, and changing that branch makes nro reject the installation. A
Git worktree avoids another full clone:

```bash
cd /path/to/nro-dev
git fetch origin
git pull --ff-only origin dev
git worktree add -b feature/example /path/to/nro-feature-example dev
cd /path/to/nro-feature-example
./install --mode branch --dev
nro branch show
```

`git worktree add` creates and checks out the Git branch. `./install` creates its
editable environment, connects it to the shared site, and registers the branch
with parent `dev`. If the branch is already registered, the installer attaches
the new checkout to its existing scientific registry. `--dev` installs test and
lint dependencies; it does not select the Git branch named `dev`.

To register explicitly, invoke `nro branch register` through an already working
nro installation:

```bash
cd /tmp
nro branch register --checkout /path/to/nro-feature-example --parent dev
```

Most users need only the installer. Explicit registration is useful when an
environment is already prepared or when a non-default parent must be recorded
before setup. A later installer recognizes the registration and attaches the
checkout. Use `nro branch attach --checkout PATH` when adding another checkout
for a branch that is already registered.

Branch names are globally reserved in the shared nro installation. Retiring a
branch does not release its name. Choose a unique, descriptive Git branch name.

### Base a branch on another development branch

The Git base and the nro parent should describe the same relationship. Register
the desired parent explicitly before installing:

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

Run `reparent` from an attached checkout of the branch being changed. It updates
artifact inheritance and reconciles affected requests. It does not change Git
history, move outputs, or merge code.

## Develop and test

Normal commands select the branch from the current working directory:

```bash
nro doctor
nro status -P nptl -p t20
nro run -P nptl -p t20 -m networks
```

The central scheduler applies the lab-wide concurrency limit. The feature branch
uses its captured code and environment, reads compatible ancestor artifacts, and
writes new outputs to its branch directory. Use `nro branch show` to confirm the
selected branch, parent, registry, and definitions store.

By default, a development branch reads the shared definitions store without
write access. Select a branch-specific definitions checkout when the feature
changes configs, workflows, models, or events:

```bash
nro branch definitions --definitions /path/to/feature-definitions
```

Return to the shared definitions with `nro branch definitions --shared`. See
[branch registration](commands/branches.md#development-definitions) for the
ownership and validation rules.

## Merge one development branch into another

The destination should be an ancestor of the source in both Git and nro. If the
merge destination changed during development, reparent the source branch and
repeat its relevant tests before opening the pull request.

Commit and publish the source branch through the normal Git workflow:

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

The Git merge makes the code available to the parent. It does not copy existing
development artifacts. If the source branch produced expensive artifacts that
remain scientifically equivalent under the merged code, preview and promote
the selected artifacts from the accepting checkout:

```bash
nro promote --from feature/example --pr CLiMBlab-Stanford/nro#123 \
  --attest-merged -P nptl -p t20 -m networks --dry-run
nro promote --from feature/example --pr CLiMBlab-Stanford/nro#123 \
  --attest-merged -P nptl -p t20 -m networks
```

Promotion is optional. Without it, the parent reuses its existing fresh outputs
and computes missing or stale work on demand. `nro promote` verifies current
contracts and copies eligible artifacts; it does not merge Git branches or
delete source files.

Before retiring the merged branch, reparent any active child branches from each
child's attached checkout:

```bash
cd /path/to/nro-child
nro branch reparent feature/child --parent dev
```

Then retire the merged branch from its own checkout:

```bash
cd /path/to/nro-feature-example
nro branch retire feature/example
```

Retirement cancels that branch's demand and prevents future checkouts. It keeps
its registry and outputs. Purge unwanted branch-owned artifacts before retiring
if they should not be retained. Git worktree and remote-branch cleanup are
separate operations and should happen only after the working tree is clean and
the merge is confirmed.

## Merge `dev` into `main`

`main` contains releases only. Every pull request into `main` must advance the
semantic version in `pyproject.toml` by at least one patch. Complete the release
notes and version change on `dev`, run the release test suite, then push `dev`:

```bash
cd /path/to/nro-dev
git fetch origin
git pull --ff-only origin dev
# Edit pyproject.toml and the applicable release documentation.
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
git tag -a v0.1.0 -m "nro 0.1.0"
git push origin v0.1.0
```

Pushing a valid version tag starts the GitHub Release workflow. Confirm that the
workflow created the corresponding GitHub Release. Do not move a published tag.

Install and activate the new release from the shared `main` checkout:

```bash
./install --maintain
nro release
nro doctor
```

The installer checks that `main` is clean, tagged, and present on `origin/main`.
It records the release and activates that checkout for future scheduler work. If
work is active, it asks whether to drain the shared pool before continuing. No
separate branch-registration or release-approval command is needed.

After activation, equivalent `dev` artifacts may be promoted with the accepted
pull-request reference:

```bash
nro promote --from dev --pr CLiMBlab-Stanford/nro#124 --attest-merged \
  -P nptl -p t20 -m networks --dry-run
```

Finally, fast-forward the dedicated `dev` checkout to the release commit and
push it so new feature work starts from the current spine:

```bash
cd /path/to/nro-dev
git fetch origin
git merge --ff-only origin/main
git push origin dev
```

Do not retire `dev`; it remains the integration branch beneath `main` in nro's
fixed branch topology.

## Command responsibilities

Git and nro track different state:

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
