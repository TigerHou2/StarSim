# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'StarSim'
copyright = '2025, Tiger Hou'
author = 'Tiger Hou'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

extensions = [
    'numpydoc',
    'nbsphinx',
    # 'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx'
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'astropy': ('https://docs.astropy.org/en/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None)
}

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "member-order": "groupwise"
}
autodoc_typehints = "description"

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']
html_css_files = [
    'css/custom.css',
]
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/TigerHou2/StarSim",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        }
    ]
}