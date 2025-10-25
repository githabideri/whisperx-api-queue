"""Namespace package that proxies the top-level modules for whisperx_api_queue.

This allows tests and consumers to use the conventional
`import whisperx_api_queue.<module>` form without moving the module files.
"""

from __future__ import annotations

import importlib
import pathlib
import sys
from types import ModuleType
from typing import Dict

_PKG_ROOT = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _PKG_ROOT.parent

# Ensure the project root (where api.py, common.py, etc. live) is importable.
_project_root_str = str(_PROJECT_ROOT)
if _project_root_str not in sys.path:
    sys.path.insert(0, _project_root_str)

_EXPORTED_MODULES = ("dependencies",)
_loaded_modules: Dict[str, ModuleType] = {}


def _load_module(name: str) -> ModuleType:
    module = importlib.import_module(name)
    fq_name = f"{__name__}.{name}"
    sys.modules[fq_name] = module
    globals()[name] = module  # type: ignore[assignment]
    _loaded_modules[name] = module
    return module


for _module_name in _EXPORTED_MODULES:
    _load_module(_module_name)


__all__ = list(_EXPORTED_MODULES)
