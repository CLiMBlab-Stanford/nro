# Contributing

These rules apply to everyone who contributes to nro, including people working
with AI tools. They cover how to disclose AI assistance, which identities may
appear on commits and pull requests, how to write documentation, and how
releases are made. The [development guide](docs/development.md) covers tests
and tooling.

## AI-assisted contributions

Disclose material AI assistance in every affected commit. Add one trailer for
each tool and model combination:

    Assisted-by: <tool-or-agent>:<model-identifier>

Examples:

    Assisted-by: Codex:gpt-5.6
    Assisted-by: Claude Code:claude-opus-4.6

Use the most specific model identifier that the tool actually exposes. If the
tool does not disclose its underlying model, use the visible product or tool
label and do not guess a model.

The human contributor remains the commit author and is responsible for
reviewing and testing the contribution. Do not list an AI system in
`Signed-off-by:`. Do not rewrite published history solely to add missing
trailers.

These trailers are a project convention written in standard Git trailer
syntax. Place them after a blank line at the end of the commit message, and keep
any existing `Signed-off-by:`, `Co-authored-by:`, issue, or review trailers.

## Commit and pull-request identity

An agent may create a commit or pull request only when the requesting human has
explicitly authorized that publication action. Having technical access does not
count as authorization for the task.

A commit may carry the human's name and email only when the user's existing Git
configuration already supplies them. A pull request may use the human's account
only when the authenticated hosting session clearly belongs to that person.

Before committing, an agent must verify that both `user.name` and `user.email`
are configured. Before creating or updating a pull request, it must verify the
identity associated with the authenticated hosting session. An agent must not:

- invent a name or email address;
- set or override identity merely to make a commit appear human-authored;
- fall back to an agent, bot, service, or different user's identity without
  explicit authorization;
- conceal bot or service attribution when such an identity is deliberately
  used; or
- request, print, store, or copy authentication secrets into the repository.

If the required identity or authenticated session is missing, ambiguous, or
unusable, the agent must stop before committing or publishing. It should leave
the changes ready for the human and provide the proposed commit message,
`Assisted-by:` trailer, pull-request title, and pull-request body as applicable.
The human can then configure Git, authenticate through the hosting tool, or
perform the publication directly.

Cryptographic commit signing must use the human's configured signing mechanism,
and an agent must never fabricate a signature. Add `Signed-off-by:` only when
the human has authorized the certification it represents.

The description of a pull request with material AI assistance should disclose
that assistance, even when its commits carry `Assisted-by:` trailers.

## AI-assisted documentation

Documentation, docstrings, and explanatory code comments created or revised
with AI assistance must follow [WRITING_POLICY.md](WRITING_POLICY.md). Apply the
policy to the text your contribution changes. Unrelated existing documentation
does not need to be rewritten.

Check technical claims against the code and supplied source material. AI
disclosure belongs in `AI_PROVENANCE.md`, commit trailers, and pull-request
metadata, not in ordinary technical documentation.

## Releases and compatibility

`main` contains released code. Prepare changes on `dev` or a feature branch,
then merge them into `main` through a reviewed pull request.

Every pull request to `main` must change the version in `pyproject.toml` to a
later `MAJOR.MINOR.PATCH` value, and the smallest permitted increment is one
patch. A repository check rejects a pull request that does not advance the
version. The required quality checks must also pass: the complete test suite,
formatting and lint checks, the documentation build, and an isolated upgrade
rehearsal.

Before pushing, run the focused tests and checks for the files and behavior you
changed. The pull-request checks run the complete suite in a clean environment
and are the authoritative release gate. You do not need to repeat that full run
locally unless the change is broad or the focused checks reveal a wider risk.

Use patch releases for compatible fixes. During the 0.x series, use minor releases
for new features and intentional interface changes. Compatibility support is welcome
when it helps current users without materially increasing complexity, runtime,
maintenance cost, or ambiguity. Document intentional incompatibilities and migration
steps in the pull request.

The upgrade rehearsal is `./install --rehearse-upgrade`. It tests maintenance
from the target release to the candidate, using temporary installation and
registry state. Run it locally when installation, orchestration schemas, migrations, or
release management changed. It does not alter the active shared installation.

After the merge, tag the release as `vMAJOR.MINOR.PATCH` and push the tag. Do
not move or replace an existing release. The tag workflow checks the package
version and its ancestry on `main`, then creates the corresponding GitHub
Release.

After the workflow succeeds, edit the GitHub Release so that it begins with a
concise, human-written summary of the important changes. Generated notes may
follow the summary, but they do not replace it. Publication is complete only
after you have verified both the workflow and the release description.

Git versions record source releases. They play no part in scientific artifact
freshness. See the [release guide](docs/commands/releases.md) for publication
and deployment.
