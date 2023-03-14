# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import datetime

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))


# -- Project information -----------------------------------------------------

year = datetime.date.today().year
project = "EPI"
copyright = f"2022-{year}, Vincent Wagner, Sebastian Höpfl, Lars Kaiser"
author = "Vincent Wagner, Sebastian Höpfl, Lars Kaiser"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",  # Needed to include the README.md in rst
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",  # Somehow sphinx-apidoc generates files which need this extension
    "sphinx.ext.githubpages",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
]


myst_enable_extensions = [
    "tasklist",
]

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# Currently used only for sphinx_copybutton
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# How to parse files depending on their ending
source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

# Fix cross-referencing in markdown files
myst_heading_anchors = 3


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
# html_theme = "sphinx_book_theme"

html_logo = "../../epi.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

todo_include_todos = 1

html_theme_options = {
    "navigation_depth": -1,
}

if html_theme == "sphinx_book_theme":
    # Add searchbar to sidebar in book theme
    html_sidebars = {
        "**": [
            "navbar-logo.html",
            "search-field.html",
            "sbt-sidebar-nav.html",
        ]  # "sidebar-nav-bs"]
    }
    # Set book theme options
    html_theme_options = {
        "repository_url": "https://github.com/Systems-Theory-in-Systems-Biology/EPI",
        "use_repository_button": True,
        "show_navbar_depth": 1,
        "show_toc_level": 2,
        "use_download_button": True,
        "home_page_in_toc": True,
        # "use_source_button": True,
        # "repository_branch": "gh-pages",
        # "use_edit_page_button": True,
    }

# Configurate autodoc
autodoc_member_order = "groupwise"
add_module_names = False
