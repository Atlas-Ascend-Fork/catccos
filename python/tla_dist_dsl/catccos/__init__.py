"""CATCCOS Python namespace."""

from __future__ import annotations

import sys
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec, PathFinder
from pathlib import Path
from types import ModuleType


class _PinnedCatlassFinder(MetaPathFinder):
    """Resolve only Catlass from the pinned tree, leaving other imports alone."""

    def __init__(self, source_root: Path) -> None:
        self.source_root = source_root

    def find_spec(self, fullname: str, path: object = None, target: ModuleType | None = None) -> ModuleSpec | None:
        if fullname != "catlass":
            return None
        return PathFinder.find_spec(fullname, [str(self.source_root)])


def _bootstrap_source_tree() -> None:
    """Prefer the pinned Catlass source tree without changing its bridge."""
    tla_dist_root = Path(__file__).resolve().parents[1]
    catccos_root = tla_dist_root.parents[1]
    pinned_tla = catccos_root / "3rdparty" / "catlass" / "python" / "tla_dsl"
    if (pinned_tla / "catlass" / "__init__.py").is_file():
        finder = _PinnedCatlassFinder(pinned_tla)
        try:
            path_finder_index = sys.meta_path.index(PathFinder)
        except ValueError:
            sys.meta_path.append(finder)
        else:
            # Keep existing loaders ahead of the filesystem package lookup.
            sys.meta_path.insert(path_finder_index, finder)


_bootstrap_source_tree()

__all__ = ["tla_dist"]
