"""Adapter discovery for the common Chinese-Jev data pipeline.

Adapter files are executable Python plugins and therefore must be trusted.  Each
plugin exports ``ADAPTERS: dict[str, callable]``.  Adapter callables use the
same interface as :mod:`chinese_jev_data_pipeline.core`:

    adapter(source, paths, audit, review) -> Iterable[DecisionCase]
"""
from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

BUILTIN_MODULES = (
    "domain_medical",
    "medical_extended",
    "medical_matching_sources",
    "medical_exam_sources",
    "medical_clinical_sources",
    "medical_additional_sources",
    "domain_legal",
    "domain_everyday",
)

def _merge(target, incoming, origin):
    if not isinstance(incoming, dict) or not incoming:
        raise ValueError(f"{origin} must expose a nonempty ADAPTERS dict")
    overlap = set(target) & set(incoming)
    if overlap:
        raise ValueError(f"duplicate adapter names from {origin}: {sorted(overlap)}")
    if any(not isinstance(name, str) or not name or not callable(fn)
           for name, fn in incoming.items()):
        raise ValueError(f"{origin} has an invalid ADAPTERS entry")
    target.update(incoming)


def _file_module(path, index):
    name = f"_chinese_jev_data_adapter_{index}_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load adapter file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def registry(config_path, config):
    """Return the merged adapter registry and all implementation files to hash."""
    adapters, files = {}, []
    for name in BUILTIN_MODULES:
        module = importlib.import_module("." + name, __package__)
        _merge(adapters, module.ADAPTERS, name)
        files.append(module.__file__)

    modules = config.get("adapter_modules", [])
    adapter_files = config.get("adapter_files", [])
    if not isinstance(modules, list) or not all(isinstance(v, str) and v for v in modules):
        raise ValueError("adapter_modules must be a list of importable module names")
    if not isinstance(adapter_files, list) or not all(isinstance(v, str) and v for v in adapter_files):
        raise ValueError("adapter_files must be a list of Python paths")

    for name in modules:
        module = importlib.import_module(name)
        _merge(adapters, getattr(module, "ADAPTERS", None), name)
        if getattr(module, "__file__", None):
            files.append(module.__file__)

    base = Path(config_path).resolve().parent
    for index, raw in enumerate(adapter_files):
        path = (base / raw).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        module = _file_module(path, index)
        _merge(adapters, getattr(module, "ADAPTERS", None), str(path))
        files.append(path)
    return adapters, list(dict.fromkeys(str(Path(p).resolve()) for p in files))
