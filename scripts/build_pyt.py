"""Bundle the package into one ArcGIS Pro toolbox file: arcgis/CenterlineAlign.pyt

The .pyt needs only what ships with ArcGIS Pro (arcpy, numpy, scipy), so it can be
copied onto a locked-down machine as a single text file.

    python scripts/build_pyt.py          # rebuild after changing the package
    python scripts/build_pyt.py --check  # fail if the committed .pyt is out of date
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "centerline_align"
OUT = ROOT / "arcgis" / "CenterlineAlign.pyt"
MODULES = ["config", "chip", "geom", "features", "classifier", "refine", "topology",
           "engine", "arcgis_impl", "arcgis_toolbox"]

HEADER = '''# -*- coding: utf-8 -*-
"""Centerline Alignment - ArcGIS Pro Python toolbox.

GENERATED FILE - do not edit by hand. Built by scripts/build_pyt.py from the
centerline_align package (version {version}).

Add to ArcGIS Pro: Catalog pane > Toolboxes > right-click > Add Toolbox, pick this
file. Needs nothing beyond ArcGIS Pro's own Python (arcpy, numpy, scipy).
"""
from __future__ import annotations

import sys as _sys
import types as _types

# dataclasses look up the defining module in sys.modules; ArcGIS may load a .pyt
# without registering it there, so stand in a placeholder while classes are built.
_placeholder = None
if __name__ not in _sys.modules:
    _placeholder = _sys.modules[__name__] = _types.ModuleType(__name__)
'''

FOOTER = '''

if _placeholder is not None and _sys.modules.get(__name__) is _placeholder:
    del _sys.modules[__name__]
'''


def strip(src: str) -> str:
    out, skipping = [], False
    for line in src.splitlines():
        if skipping:
            skipping = ")" not in line
            continue
        if line.startswith("from __future__"):
            continue
        if re.match(r"from \.[\w.]* import", line) or line.startswith("from . import"):
            skipping = line.rstrip().endswith("(")
            continue
        out.append(line)
    return "\n".join(out).strip() + "\n"


def build() -> str:
    version = re.search(r'__version__ = "([^"]+)"', (PKG / "__init__.py").read_text()).group(1)
    parts = [HEADER.format(version=version)]
    for m in MODULES:
        parts.append(f"\n# {'=' * 70}\n# {m}.py\n# {'=' * 70}\n")
        parts.append(strip((PKG / f"{m}.py").read_text()))
    parts.append(FOOTER)
    return "\n".join(parts)


if __name__ == "__main__":
    text = build()
    if "--check" in sys.argv:
        if not OUT.exists() or OUT.read_text() != text:
            sys.exit(f"{OUT.relative_to(ROOT)} is out of date; run python scripts/build_pyt.py")
        print("up to date")
    else:
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(text)
        print(f"wrote {OUT.relative_to(ROOT)} ({len(text.splitlines())} lines)")
