# Repository instructions for AI agents

## AI provenance

Before making a material change to this repository, read
`AI_PROVENANCE.md` and register the provider, tool or agent, exposed model
identifier, period, and roles under **Systems used**. If the same system is
already registered, update its period or roles only when needed; do not add a
duplicate row for each session.

Use only model identity information explicitly exposed by the environment.
Never infer a version or store prompts, secrets, proprietary context,
conversation logs, or session identifiers.

AI systems must not be recorded as Git authors, copyright holders, DCO
signatories, or `Signed-off-by:` identities. The human contributor owns and
approves the change. When handing off work intended for a commit, remind the
human contributor of the `Assisted-by:` convention in `CONTRIBUTING.md`; do not
rewrite or amend history merely to add attribution.

## Commits and pull requests

Commit, push, or create or update a pull request only when the user explicitly
requests that publication action. Before committing, verify that the existing
Git configuration supplies both the requesting human's `user.name` and
`user.email`. Before using a hosting CLI or API, verify that its authenticated
session clearly belongs to the requesting human. Do not change identity settings
or use `--author` merely to make an agent-created commit appear human-authored.

If identity is missing or ambiguous, authentication is unavailable, or the
available session belongs to another user, bot, or service, stop before the
publication action and explain the problem. Never silently fall back to
`codex`, another agent identity, or another available account. A bot or service
account may be used only when the user explicitly authorizes it, and its real
attribution must remain visible.

Do not ask the user to send authentication secrets and never print, store, or
copy credentials into the repository. Instead, ask the user to configure Git,
authenticate through the relevant tool, or publish the prepared work directly.
When blocked, leave the working tree ready and provide the proposed commit
message with its `Assisted-by:` trailer and any proposed pull-request title and
body.

Never fabricate a cryptographic signature. Use a configured human signing
mechanism only with authorization. Add `Signed-off-by:` only when the human has
authorized the certification it represents. Follow the complete policy in
`CONTRIBUTING.md`.

## Writing

Before creating or revising documentation, docstrings, or explanatory code
comments, read and follow `WRITING_POLICY.md`. Treat it as the authoritative
repository style guide for agent-generated documentation. Apply it to every
piece of prose touched by the task, but do not rewrite unrelated text merely to
make it conform.

Check technical claims against the implementation and nearby documentation.
Use the repository's established terms and keep comments and docstrings aligned
with the code. AI disclosure belongs in `AI_PROVENANCE.md` and commit or
pull-request metadata, not in ordinary technical documentation.
