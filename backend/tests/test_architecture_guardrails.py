"""Lightweight import guardrails for modular backend layering."""

from __future__ import annotations

import ast
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1] / "app"


def _iter_python_files(layer: str) -> list[Path]:
    return sorted((APP_ROOT / layer).rglob("*.py"))


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
    return imports


def _assert_no_forbidden_imports(layer: str, forbidden_prefixes: tuple[str, ...]) -> None:
    violations: list[str] = []
    for path in _iter_python_files(layer):
        modules = _imported_modules(path)
        for module in sorted(modules):
            if module.startswith(forbidden_prefixes):
                rel_path = path.relative_to(APP_ROOT.parent)
                violations.append(f"{rel_path}: {module}")
    assert not violations, "Forbidden imports detected:\n" + "\n".join(violations)


def test_api_layer_does_not_import_pipeline_or_infrastructure() -> None:
    _assert_no_forbidden_imports("api", ("app.pipeline", "app.infrastructure"))


def test_application_layer_does_not_import_api_layer() -> None:
    _assert_no_forbidden_imports("application", ("app.api",))


def test_pipeline_layer_does_not_import_api_or_application_layers() -> None:
    _assert_no_forbidden_imports("pipeline", ("app.api", "app.application"))


def test_infrastructure_layer_does_not_import_upper_layers() -> None:
    _assert_no_forbidden_imports(
        "infrastructure",
        ("app.api", "app.application", "app.pipeline"),
    )
