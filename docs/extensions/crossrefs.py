"""Griffe extension that resolves IDE-friendly docstring links as cross-references.

Docstrings link with inline Markdown, ``[`Objective`](gpc.objectives.Objective)``, which
renders in editor hovers. For the docs, these are rewritten into mkdocstrings
cross-references, ``[`Objective`][gpc.objectives.Objective]``.
"""

import re
from typing import Any

import griffe

_LINK = re.compile(r"\]\((gpc(?:\.\w+)*)\)")


def _rewrite(obj: griffe.Object) -> None:
    if obj.docstring is not None:
        obj.docstring.value = _LINK.sub(r"][\1]", obj.docstring.value)


class Crossrefs(griffe.Extension):
    def on_object(self, *, obj: griffe.Object, **kwargs: Any) -> None:
        _rewrite(obj)

    # The root package does not trigger on_object.
    def on_package(self, *, pkg: griffe.Module, **kwargs: Any) -> None:
        _rewrite(pkg)
