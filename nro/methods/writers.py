"""Write a methods document as RTF or as a LaTeX fragment."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence

from nro.methods.bibliography import CitationLabels
from nro.methods.document import MethodsDocument, Paragraph, Run, Section, Table

FORMATS = ("rtf", "latex")
EXTENSIONS = {"rtf": "rtf", "latex": "tex"}


def _grouped(document: MethodsDocument) -> Iterator[tuple[str, list[Section]]]:
    current: str | None = None
    group: list[Section] = []
    for section in document.sections:
        if section.module != current and group:
            yield current or "", group
            group = []
        current = section.module
        group.append(section)
    if group:
        yield current or "", group


# RTF


def _rtf_escape(text: str) -> str:
    pieces: list[str] = []
    for character in text:
        code = ord(character)
        if character in "\\{}":
            pieces.append("\\" + character)
        elif character == "\n":
            pieces.append("\\line ")
        elif character == "\u00a0":
            pieces.append("\\~")
        elif code < 128:
            pieces.append(character)
        else:
            encoded = character.encode("utf-16-le")
            for index in range(0, len(encoded), 2):
                unit = int.from_bytes(encoded[index : index + 2], "little", signed=True)
                pieces.append(f"\\u{unit}?")
    return "".join(pieces)


def _citation_text(run: Run, labels: CitationLabels) -> str:
    if run.narrative:
        return labels.narrative[run.keys[0]]
    return "(" + "; ".join(labels.parenthetical[key] for key in run.keys) + ")"


def _rtf_runs(runs: Paragraph, labels: CitationLabels) -> str:
    parts = []
    for run in runs:
        if run.kind == "text":
            parts.append(_rtf_escape(run.value))
        elif run.kind == "emphasis":
            parts.append("{\\i " + _rtf_escape(run.value) + "}")
        else:
            parts.append(_rtf_escape(_citation_text(run, labels)))
    return "".join(parts)


def _rtf_heading(text: str, size: int) -> str:
    return f"{{\\pard\\keepn\\sb240\\sa120\\b\\fs{size} {_rtf_escape(text)}\\par}}\n"


def _rtf_table(table: Table) -> str:
    width = 9360 // max(len(table.columns), 1)
    boundaries = "".join(
        f"\\clbrdrb\\brdrs\\brdrw10\\cellx{width * (index + 1)}"
        for index in range(len(table.columns))
    )
    lines = [_rtf_heading(table.title, 24)]
    for row_index, row in enumerate((table.columns, *table.rows)):
        bold = "\\b " if row_index == 0 else ""
        cells = "".join(f"\\pard\\intbl\\fs18 {bold}{_rtf_escape(cell)}\\cell " for cell in row)
        header = "\\trhdr" if row_index == 0 else ""
        lines.append(f"{{\\trowd\\trgaph72{header}{boundaries}\n{cells}\\row}}\n")
    lines.append("{\\pard\\sa120\\par}\n")
    return "".join(lines)


def write_rtf(document: MethodsDocument, labels: CitationLabels) -> str:
    """Return the document as RTF with author-year citations and references."""
    out = [
        "{\\rtf1\\ansi\\ansicpg1252\\uc1\\deff0",
        "{\\fonttbl{\\f0\\froman Times New Roman;}}",
        "\\f0\\fs24\n",
    ]
    out.append(_rtf_heading("Methods", 32))
    for module, sections in _grouped(document):
        out.append(_rtf_heading(document.module_titles.get(module, module), 28))
        for section in sections:
            out.append(_rtf_heading(section.title, 24))
            for paragraph in section.paragraphs:
                out.append(
                    f"{{\\pard\\sa160\\sl360\\slmult1 {_rtf_runs(paragraph, labels)}\\par}}\n"
                )
    if document.references:
        out.append(_rtf_heading("References", 28))
        for reference in document.references:
            out.append(f"{{\\pard\\sa120\\fi-720\\li720 {_rtf_runs(reference, labels)}\\par}}\n")
    if document.tables:
        out.append("{\\pard\\page\\par}\n")
        out.append(_rtf_heading("Supplementary tables", 28))
        out.extend(_rtf_table(table) for table in document.tables)
    out.append(f"{{\\pard\\sb240\\i\\fs18 {_rtf_escape(document.footer)}\\par}}\n")
    out.append("}\n")
    return "".join(out)


# LaTeX

_LATEX_ESCAPES = {
    "\\": "\\textbackslash{}",
    "&": "\\&",
    "%": "\\%",
    "$": "\\$",
    "#": "\\#",
    "_": "\\_",
    "{": "\\{",
    "}": "\\}",
    "~": "\\textasciitilde{}",
    "^": "\\textasciicircum{}",
    "\u00a0": "~",
}


def _latex_escape(text: str) -> str:
    return "".join(_LATEX_ESCAPES.get(character, character) for character in text)


def _latex_runs(runs: Paragraph) -> str:
    parts = []
    for run in runs:
        if run.kind == "text":
            parts.append(_latex_escape(run.value))
        elif run.kind == "emphasis":
            parts.append("\\emph{" + _latex_escape(run.value) + "}")
        elif run.narrative:
            parts.append("\\citet{" + run.keys[0] + "}")
        else:
            parts.append("\\citep{" + ",".join(run.keys) + "}")
    return "".join(parts)


def _latex_table(table: Table) -> str:
    count = max(len(table.columns), 1)
    spec = "".join(f"p{{{0.95 / count:.3f}\\linewidth}}" for _ in range(count))
    header = " & ".join(f"\\textbf{{{_latex_escape(column)}}}" for column in table.columns)
    body = "".join(
        " & ".join(_latex_escape(cell) for cell in row) + " \\\\\n" for row in table.rows
    )
    return (
        f"\\subsection*{{{_latex_escape(table.title)}}}\n"
        "{\\scriptsize\n"
        f"\\begin{{longtable}}{{{spec}}}\n\\hline\n{header} \\\\\n\\hline\n\\endhead\n"
        f"{body}\\hline\n\\end{{longtable}}\n}}\n\n"
    )


def write_latex(document: MethodsDocument, labels: CitationLabels) -> str:
    """Return a LaTeX fragment that uses natbib citations and ``longtable``.

    The caller's preamble loads ``natbib`` and ``longtable`` and supplies
    ``\\bibliographystyle`` and ``\\bibliography{references}``.
    """
    del labels
    out = [
        "% Generated by nro methods. Requires \\usepackage{natbib} and \\usepackage{longtable}.\n",
        "% Cite with an author-year style and \\bibliography{references}.\n",
        f"% {document.footer}\n\n",
        "\\section*{Methods}\n\n",
    ]
    for module, sections in _grouped(document):
        out.append(
            f"\\subsection*{{{_latex_escape(document.module_titles.get(module, module))}}}\n\n"
        )
        for section in sections:
            out.append(f"\\subsubsection*{{{_latex_escape(section.title)}}}\n\n")
            for paragraph in section.paragraphs:
                out.append(_latex_runs(paragraph) + "\n\n")
    if document.tables:
        out.append("\\section*{Supplementary tables}\n\n")
        out.extend(_latex_table(table) for table in document.tables)
    return "".join(out)


WRITERS: dict[str, Callable[[MethodsDocument, CitationLabels], str]] = {
    "rtf": write_rtf,
    "latex": write_latex,
}


def tsv(table: Table) -> str:
    """Return a supplementary table as tab-separated text with a header row."""

    def clean(cell: str) -> str:
        return cell.replace("\t", " ").replace("\n", " ")

    rows: Sequence[tuple[str, ...]] = (table.columns, *table.rows)
    return "".join("\t".join(clean(cell) for cell in row) + "\n" for row in rows)
