# llmri/cli/__init__.py

"""
CLI package for llmri.

The console entrypoint is `llmri.cli.main:main`, exposed as the `llmri` script.

We intentionally do not import `main` here to avoid double-import warnings
when running `python -m llmri.cli.main`.
"""

__all__: list[str] = []
