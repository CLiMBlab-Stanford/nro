"""Load and render the editable methods templates.

Each template is a Markdown file with YAML front matter. The body is rendered by a
sandboxed Jinja2 environment and then parsed into paragraphs of plain text,
``*emphasis*``, parenthetical citations ``[@a; @b]``, and narrative citations
``@a``. Any other Markdown is rejected so every writer can represent the result.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml
from jinja2 import StrictUndefined, TemplateSyntaxError, UndefinedError, meta
from jinja2.sandbox import ImmutableSandboxedEnvironment

from nro.methods.document import Paragraph, Run, Section

TEMPLATE_ROOT = Path(__file__).with_name("templates")
_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.DOTALL)
_TEMPLATE_NAME = re.compile(r"\A\d{2}_[a-z0-9_]+\.md\Z")
_FRONT_MATTER_KEYS = {"section", "requires", "when", "context_version"}
_INLINE = re.compile(
    r"\[(@[\w:-]+(?:\s*;\s*@[\w:-]+)*)\]"  # parenthetical citation group
    r"|(?<![\w@])@([A-Za-z][\w:-]*)"  # narrative citation
    r"|\*([^*\n]+)\*"  # emphasis
)
_UNSUPPORTED_MARKDOWN = (
    (re.compile(r"(?m)^\s*#"), "headings"),
    (re.compile(r"(?m)^\s*([-+*]|\d+\.)\s"), "lists"),
    (re.compile(r"`"), "code spans"),
    (re.compile(r"\*\*|__"), "strong emphasis"),
    (re.compile(r"\]\("), "links"),
    (re.compile(r"</?[A-Za-z]"), "HTML tags"),
)


class MethodsTemplateError(ValueError):
    """Raised when a template is malformed or does not match its context."""


def _percent(value: object, digits: int = 0) -> str:
    number = float(value)  # type: ignore[arg-type]
    return f"{_number(number * 100, digits)}%"


def _number(value: object, digits: int = 3) -> str:
    """Format a number without trailing zeros, rejecting non-finite values."""
    if isinstance(value, bool):
        raise MethodsTemplateError("A boolean cannot be formatted as a number")
    if isinstance(value, int):
        return f"{value:,}" if abs(value) >= 10000 else str(value)
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number):
        raise MethodsTemplateError(f"Cannot format non-finite number {number!r}")
    text = f"{number:.{digits}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"-0", ""} else text


def _count_of(count: object, total: object) -> str:
    return f"{int(count)} of {int(total)}"  # type: ignore[call-overload]


def _join_and(values: Iterable[object], conjunction: str = "and") -> str:
    items = [str(value) for value in values]
    if not items:
        raise MethodsTemplateError("Cannot join an empty list")
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} {conjunction} {items[1]}"
    return ", ".join(items[:-1]) + f", {conjunction} " + items[-1]


def _plural(count: object, singular: str, plural: str | None = None) -> str:
    return singular if int(count) == 1 else (plural or singular + "s")  # type: ignore[call-overload]


def environment() -> ImmutableSandboxedEnvironment:
    """Return the sandboxed Jinja2 environment shared by all methods templates."""
    env = ImmutableSandboxedEnvironment(
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=False,
    )
    env.filters.clear()
    env.filters.update(
        {
            "pct": _percent,
            "num": _number,
            "count_of": _count_of,
            "join_and": _join_and,
            "plural": _plural,
            "length": len,
        }
    )
    env.globals.clear()
    env.tests.update({"none": lambda value: value is None})
    return env


@dataclass(frozen=True)
class MethodsTemplate:
    """One parsed template file."""

    module: str
    name: str
    path: Path
    section: str
    requires: tuple[str, ...]
    when: str
    context_version: int
    body: str

    @property
    def relative_path(self) -> str:
        """Return the path used in traces, such as ``func/02_motion.md``."""
        return f"{self.module}/{self.name}"


def parse_template(module: str, path: Path) -> MethodsTemplate:
    """Parse front matter and check that ``requires`` matches the variables used."""
    if not _TEMPLATE_NAME.match(path.name):
        raise MethodsTemplateError(f"{path}: template names must look like 01_section_name.md")
    match = _FRONT_MATTER.match(path.read_text(encoding="utf-8"))
    if match is None:
        raise MethodsTemplateError(f"{path}: missing YAML front matter")
    header = yaml.safe_load(match.group(1))
    if not isinstance(header, dict) or set(header) - _FRONT_MATTER_KEYS:
        raise MethodsTemplateError(
            f"{path}: front matter may contain only {', '.join(sorted(_FRONT_MATTER_KEYS))}"
        )
    for key in ("section", "requires", "context_version"):
        if key not in header:
            raise MethodsTemplateError(f"{path}: front matter needs {key!r}")
    requires = header["requires"]
    if not isinstance(requires, list) or not all(isinstance(item, str) for item in requires):
        raise MethodsTemplateError(f"{path}: requires must be a list of context field names")
    template = MethodsTemplate(
        module=module,
        name=path.name,
        path=path,
        section=str(header["section"]),
        requires=tuple(requires),
        when=str(header.get("when", "true")),
        context_version=int(header["context_version"]),
        body=match.group(2),
    )
    env = environment()
    try:
        used = meta.find_undeclared_variables(env.parse(template.body)) | (
            meta.find_undeclared_variables(env.parse("{{ " + template.when + " }}"))
        )
    except TemplateSyntaxError as error:
        raise MethodsTemplateError(f"{path}: {error}") from error
    declared = {name.split(".", 1)[0] for name in template.requires}
    if used != declared:
        raise MethodsTemplateError(
            f"{path}: requires names {sorted(declared)} but the template uses {sorted(used)}"
        )
    return template


def module_templates(
    module: str,
    *,
    context_version: int,
    overrides: Sequence[Path] = (),
) -> tuple[MethodsTemplate, ...]:
    """Return a module's templates in section order, applying override directories.

    The first override directory that contains a file wins over later ones and over
    the packaged template. A template written for another context version fails.
    """
    selected: dict[str, Path] = {}
    for root in (TEMPLATE_ROOT, *reversed(tuple(overrides))):
        directory = Path(root) / module
        if directory.is_dir():
            for path in sorted(directory.glob("*.md")):
                selected[path.name] = path
    templates = tuple(parse_template(module, selected[name]) for name in sorted(selected))
    for template in templates:
        if template.context_version != context_version:
            raise MethodsTemplateError(
                f"{template.path}: written for {module} methods context version "
                f"{template.context_version}, but the current version is {context_version}"
            )
    return templates


def _lookup(context: Mapping[str, object], dotted: str) -> object:
    value: object = context
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise MethodsTemplateError(f"Methods context has no field {dotted!r}")
        value = value[part]
    return value


def _inline_runs(text: str, *, source: str) -> Paragraph:
    for pattern, label in _UNSUPPORTED_MARKDOWN:
        if pattern.search(text):
            raise MethodsTemplateError(f"{source}: templates may not contain {label}")
    runs: list[Run] = []
    position = 0
    for match in _INLINE.finditer(text):
        if match.start() > position:
            runs.append(Run.text(text[position : match.start()]))
        group, narrative, emphasis = match.groups()
        if group is not None:
            keys = tuple(part.strip()[1:] for part in group.split(";"))
            runs.append(Run.citation(keys))
        elif narrative is not None:
            runs.append(Run.citation((narrative,), narrative=True))
        else:
            runs.append(Run.emphasis(emphasis))
        position = match.end()
    if position < len(text):
        runs.append(Run.text(text[position:]))
    stray = [
        run.value for run in runs if run.kind == "text" and ("@" in run.value or "*" in run.value)
    ]
    if stray:
        raise MethodsTemplateError(
            f"{source}: unmatched citation or emphasis marker in {stray[0]!r}"
        )
    return tuple(runs)


def render_template(template: MethodsTemplate, context: Mapping[str, object]) -> Section | None:
    """Render one template, or return None when its ``when`` condition is false."""
    for name in template.requires:
        _lookup(context, name)
    env = environment()
    try:
        enabled = env.compile_expression(template.when, undefined_to_none=False)(**context)
        if not enabled:
            return None
        # Paragraphs are separated by blank lines in the template source. Mark them
        # before rendering so lines left empty by block tags do not split paragraphs.
        source = re.sub(r"\n[ \t]*\n\s*", "\n\u2029\n", template.body.strip())
        rendered = env.from_string(source).render(**context)
    except (UndefinedError, TemplateSyntaxError, TypeError, ValueError) as error:
        raise MethodsTemplateError(f"{template.path}: {error}") from error
    blocks = [re.sub(r"\s+", " ", block).strip() for block in rendered.split("\u2029")]
    paragraphs = tuple(
        _inline_runs(block, source=template.relative_path) for block in blocks if block
    )
    if not paragraphs:
        return None
    return Section(
        module=template.module,
        title=template.section,
        paragraphs=paragraphs,
        template=template.relative_path,
    )
