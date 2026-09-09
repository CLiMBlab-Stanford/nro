"""Build documentation from source without importing neuroimaging dependencies."""

import tomllib
from pathlib import Path

project = "nro"
author = "nro contributors"
release = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())[
    "project"
]["version"]
version = release
extensions = ["myst_parser", "autoapi.extension", "sphinx.ext.napoleon", "sphinx.ext.mathjax"]
html_theme = "sphinx_rtd_theme"
exclude_patterns = ["_build", "requirements.txt"]
myst_enable_extensions = ["colon_fence", "dollarmath"]
myst_heading_anchors = 3
autoapi_dirs = [str(Path(__file__).resolve().parents[1] / "nro")]
autoapi_type = "python"
autoapi_options = ["members", "undoc-members", "show-inheritance", "show-module-summary"]
autoapi_python_class_content = "both"
autoapi_add_toctree_entry = False
autoapi_member_order = "bysource"
html_static_path = []
