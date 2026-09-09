# Contributing

## AI-assisted contributions

Disclose material AI assistance in every affected commit by adding one trailer
for each tool/model combination:

    Assisted-by: <tool-or-agent>:<model-identifier>

Examples:

    Assisted-by: Codex:gpt-5.6
    Assisted-by: Claude Code:claude-opus-4.6

Use the most specific model identifier actually exposed by the tool. If the
underlying model is not disclosed, use the visible product or tool label without
guessing a model.

The human contributor remains the commit author and is responsible for
reviewing and testing the contribution. Do not list an AI system in
`Signed-off-by:`. Do not rewrite published history solely to add missing
trailers.

These are project conventions implemented with standard Git trailer syntax.
Place trailers after a blank line at the end of the commit message, preserving
any existing `Signed-off-by:`, `Co-authored-by:`, issue, or review trailers.

## Commit and pull-request identity

An agent may create a commit or pull request only when the requesting human has
explicitly authorized that publication action. The resulting commit may use the
human's name and email only when they are already available from the user's Git
configuration. The pull request may use the human's account only when the
available authenticated hosting session clearly belongs to that person.
Technical access alone does not replace task-specific authorization.

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

If the required identity or authenticated session is absent, ambiguous, or
unusable, the agent must stop before committing or publishing. It should leave
the changes ready for the human and provide the proposed commit message,
`Assisted-by:` trailer, pull-request title, and pull-request body as applicable.
The human can then configure Git, authenticate through the hosting tool, or
perform the publication directly.

Cryptographic commit signing must use the human's configured signing mechanism.
An agent must never fabricate a signature. `Signed-off-by:` may be added only
when the human has authorized the certification it represents. Pull-request
descriptions for materially AI-assisted changes should disclose that assistance
even when the commits contain `Assisted-by:` trailers.

## AI-assisted documentation

Documentation, docstrings, and explanatory code comments created or revised
with AI assistance must follow [WRITING_POLICY.md](WRITING_POLICY.md). Apply the
policy to text changed by the contribution; unrelated existing documentation
does not need to be rewritten.

Review technical claims against the code and supplied source material. Keep AI
disclosure in `AI_PROVENANCE.md`, commit trailers, and pull-request metadata
rather than inserting it into ordinary technical documentation.

## Releases and compatibility

`main` contains released code. Prepare changes on `dev` or a feature branch and merge
them into `main` through a reviewed pull request. Every pull request to `main` must
change the version in `pyproject.toml` to a later `MAJOR.MINOR.PATCH` value. The
smallest permitted increment is one patch version. A repository check rejects a pull
request that does not advance the version.

Use patch releases for compatible fixes. During the 0.x series, use minor releases
for new features and intentional interface changes. Compatibility support is welcome
when it helps current users without materially increasing complexity, runtime,
maintenance cost, or ambiguity. Document intentional incompatibilities and migration
steps in the pull request.

After merge, tag the release as `vMAJOR.MINOR.PATCH` and push the tag without moving
or replacing an existing release. The tag workflow checks
the package version and main ancestry, then creates the corresponding GitHub Release.
Verify that the workflow succeeded before considering publication complete. Git
versions record source releases; they do not participate in scientific artifact
freshness. See the [release guide](docs/commands/releases.md) for publication and
deployment.
