# Writing policy

This policy applies when an AI agent creates or revises documentation,
docstrings, or explanatory code comments in this repository. Write for other
engineers. The text must be technically accurate, concise, natural, and easy to
scan.

## Writing principles

* Write in plain, direct language.
* Prefer short sentences and familiar words.
* Use concrete verbs and specific nouns.
* Explain concepts in the order a reader needs them.
* State the main point early, then add supporting detail.
* Match the project’s existing terminology, style, and documentation conventions.
* Assume the reader is capable but may be unfamiliar with this part of the codebase.
* Include enough context for the text to be useful without requiring the reader to inspect the implementation.
* Preserve important qualifications, constraints, failure modes, and edge cases.
* Use examples when they clarify behavior faster than prose.
* Keep established technical terms when they are more precise than a plain-language substitute.

## Avoid

* Jargon that does not improve precision.
* Marketing language, hype, cheerleading, and exaggerated claims.
* Unnecessary adjectives and adverbs.
* Long noun phrases. Rewrite them as shorter clauses or sentences.
* New terminology for ideas that already have standard names.
* Metaphors or cute labels that readers must learn.
* Repetition, throat-clearing, and summaries that merely restate nearby text.
* Claims that something is “simple,” “easy,” “obvious,” or “intuitive.”
* Stock AI phrasing such as “genuinely,” “seamlessly,” “robust,” “powerful,” “comprehensive,” “elegant,” “delve,” “leverage,” “unlock,” “ensure,” or “serves as,” unless the word is necessary and precise.
* Contrast formulas such as “It’s not X; it’s Y,” “This isn’t just X,” or “Rather than merely X, it Y.”
* Empty transition phrases such as “It is important to note that,” “At its core,” “In essence,” “That said,” and “In today’s…”
* Excessive headings, bullet lists, parenthetical remarks, em dashes, and rhetorical questions.
* Talking about the writing process or announcing what the documentation will explain.
* Mentioning that the text was generated or revised by AI. The project handles disclosure separately.

## Documentation requirements

* Describe observable behavior before implementation details.
* Explain purpose, inputs, outputs, side effects, errors, and important constraints where relevant.
* Use exact names for functions, parameters, types, commands, configuration keys, and files.
* Do not infer behavior that the code does not establish.
* Do not copy the function name into prose and treat that as an explanation.
* Do not document self-evident syntax. Focus on intent, contracts, surprising behavior, and decisions a caller must make.
* Keep code examples minimal, valid, and consistent with the current API.
* Separate distinct ideas into paragraphs instead of packing them into one dense sentence.
* Use lists only when the items are genuinely parallel or easier to scan as a list.

## Docstring requirements

* Follow the language and repository’s established docstring format.
* Begin with a brief, active-voice summary of what the code does.
* Add detail only when it helps a caller use the API correctly.
* Document parameters by meaning and constraints, not by repeating their names or types.
* Document return values when their meaning is not obvious.
* Document raised errors that callers may need to handle.
* Mention mutations, I/O, caching, ordering, thread safety, units, time zones, ownership, or lifecycle rules when relevant.
* Do not describe private implementation steps unless they affect the contract.
* Do not add sections that would be empty or redundant.
* Keep comments and docstrings synchronized with the code.

## Method

1. Read the relevant code and nearby documentation before writing.
2. Identify the intended audience and the facts the reader needs.
3. Draft the smallest complete explanation.
4. Check every technical claim against the code or supplied source material.
5. Edit for natural rhythm, brevity, and clear sentence structure.
6. Remove generic filler and wording that could apply to any project.
7. Confirm that terminology and formatting match the repository.

