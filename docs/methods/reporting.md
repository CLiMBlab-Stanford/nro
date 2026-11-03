# Methods reporting

`nro methods` writes a methods section for the module lineages that a workflow
selects. The text has author-year citations and a reference list, and it comes
with a BibTeX file and supplementary tables. It is assembled from editable
templates and the recorded configuration and derivative manifests. The same
inputs always produce the same files.

Templates currently exist for `anat`, `func`, and `clean`. Requesting another
module, or a module whose upstream lineage includes one without templates, fails
with an error.

## Running the command

```bash
nro methods main -P PROJECT --format rtf -o methods/
nro methods main -P hcp moth --task rest --format rtf -o methods/
nro methods main --preview --format latex -m func
```

`WORKFLOW` names the workflow to describe. `--format` is required:

* `rtf` writes a document that opens in Word and imports into Google Docs. It
  contains author-year in-text citations, a reference list, and the
  supplementary tables.
* `latex` writes a fragment for `\input`. It cites with natbib `\citep` and
  `\citet` and prints the tables with `longtable`. Your preamble loads both
  packages and supplies `\bibliographystyle` and `\bibliography{references}`.

Give exactly one of `-P PROJECT` or `--preview`:

* With `-P`, nro reads the registry's recorded artifact states for each named
  project. Several projects are pooled into one report, so preprocessing that
  they share is described once. Only fresh artifacts contribute. The prose reports counts such as how many runs
  used each distortion-correction method. Stale, missing, and unreadable work
  items are left out and counted on stderr and in `methods.json`. The command
  does not reassess freshness, so run `nro status --update` first if the
  registry may be out of date. A development installation cannot read the
  production registry and must use `--preview`.
* `--preview` describes the configuration alone. It does not need a registry or
  any derivatives.

With `-P`, you can narrow the counted work items with the selectors `nro run`
uses. `-p ID` selects participants, `-r ENTITY=VALUE[,VALUE...]` selects BIDS
runs, and `--task TASK` selects source tasks. Values match exactly.

Some modules' work items do not record a selected entity. Anatomical work items,
for example, have no task. Such a module is limited to the participants who have
a fresh, selected work item in a reported module that records every selected
entity. When both modules record sessions, the match is by session. So
`--task rest` describes the anatomy of exactly the sessions whose rest runs are
reported. If no reported module records the selected entities, as with
`-m anat --task rest`, the command fails.

The selection is recorded under `scope` in `methods.json`.

`-m MODULE` limits the report to that module and its upstream lineages. You can
repeat it. Without `-m`, every module with templates is described. `-o DIR`
sets the output directory, which defaults to the current directory. Files of the
same names are replaced.

Reading manifests from network storage takes time. A project with about 500 BOLD
runs took about 30 seconds on a cold cache.

## Output files

| File | Contents |
| --- | --- |
| `methods.rtf` or `methods.tex` | Methods prose, citations, references (RTF only), and supplementary tables. |
| `references.bib` | Exactly the entries the rendered text cites. |
| `supplementary_runs.tsv` | One row per run-level fact, with the project and BIDS entities, such as the realized distortion-correction method or a fallback reason. |
| `supplementary_software.tsv` | One row per tool and version. |
| `methods.json` | The lineages, template files, context values, and citations behind each section. |

The prose does not state software versions. The software table lists the
container images each module ran, the tool versions recorded in manifests, and
the nro and Python package versions of the installation that wrote the report.
An image version is listed only when the site image's acquisition receipt names
the pinned source. Otherwise the version is `unknown`. Rows whose `source` is
`reporting installation` describe the installation running `nro methods`,
which may differ from the one that produced the derivatives.

A citation that appears only in an omitted section is not included. For
example, the ICA-AROMA reference appears only when the workflow uses ICA-AROMA.

## Templates

Each module's templates live in `nro/methods/templates/MODULE/`. Files are
named `NN_section.md`, and the number sets the section order. A template has
YAML front matter and a Markdown body:

```markdown
---
section: Motion outliers
context_version: 1
when: "confounds.fd_threshold_mm > 0"
requires: [confounds.fd_threshold_mm]
---
Frames with framewise displacement greater than
{{ confounds.fd_threshold_mm | num }} mm [@power2012spurious] were marked.
```

* `section` is the heading.
* `context_version` must equal the module's current methods-context version.
* `when` is an optional Jinja expression. When it is false, the section is left
  out.
* `requires` lists the context fields that the body and `when` read. A template
  that uses a field it does not list, or lists a field it does not use, fails.

The body is rendered by a sandboxed Jinja2 environment. Undefined fields raise
an error instead of rendering blank text. The available filters are `num`,
`pct`, `count_of`, `join_and`, `plural`, and `length`. A blank line in the
template source starts a new paragraph. Other line breaks, including the lines
left by `{% if %}` tags, are joined with spaces.

The body may contain plain text, `*emphasis*`, parenthetical citations such as
`[@a]` or `[@a; @b]`, and narrative citations such as `@a`. Every key must
exist in `nro/methods/references.bib`. Headings, lists, links, code spans, and
HTML are rejected so that both output formats can represent every template.

### Context fields

Templates read a methods context, never raw configuration keys. Each module
builds its context in `nro/modules/MODULE/methods.py` and declares the field
types in `CONTEXT_FIELDS`. The context combines configured values with counts
aggregated from fresh manifests. `observed` is false in a preview, and the
templates then omit counts.

Every substantive configuration parameter must either feed a context field,
through `CONFIGURATION_FIELDS`, or appear in `METHODS_SILENT` with a reason. The
methods test suite enforces this, so a new scientific option cannot ship
without methods text or an explicit decision to omit it.

### Overrides

`--templates DIR` adds an override directory laid out like the packaged
templates, for example `DIR/func/04_distortion.md`. A file there replaces the
packaged file with the same name, and a new name adds a section. When you give
several directories, earlier ones win. An override must declare the current
`context_version`. After an nro upgrade changes a module's context, an outdated
override fails until you update it.
