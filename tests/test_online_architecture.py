"""Regression tests for the deployed ML service ownership boundary."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from api.main import app, require_internal_secret


ROOT = Path(__file__).resolve().parents[1]
ONLINE_MODULES = (
    ROOT / "api" / "main.py",
    ROOT / "feedback" / "event_handlers.py",
    ROOT / "feedback" / "producer.py",
    ROOT / "feedback" / "consumer.py",
    ROOT / "retrieval_engine.py",
    ROOT / "retrieval" / "candidate_retriever.py",
)
SQL_PATTERN = re.compile(
    r"\b(?:SELECT\s+.+?\s+FROM|INSERT\s+INTO|UPDATE\s+\w+\s+SET|DELETE\s+FROM|CREATE\s+TABLE)\b",
    re.IGNORECASE | re.DOTALL,
)


def test_online_modules_do_not_import_app_database():
    violations: list[str] = []
    for path in ONLINE_MODULES:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            if any(module == "database" or module.startswith("database.") for module in modules):
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")

    assert violations == []


def test_online_modules_contain_no_sql_queries():
    violations = [
        str(path.relative_to(ROOT))
        for path in ONLINE_MODULES
        if SQL_PATTERN.search(path.read_text())
    ]
    assert violations == []


def test_online_modules_have_no_reverse_backend_dependency():
    violations = [
        str(path.relative_to(ROOT))
        for path in ONLINE_MODULES
        if "BACKEND_URL" in path.read_text()
    ]
    assert violations == []


def test_every_non_health_api_route_requires_internal_secret():
    unprotected: list[str] = []
    for route in app.routes:
        path = getattr(route, "path", "")
        if not path.startswith("/api/v1/") or path == "/api/v1/health":
            continue
        dependencies = {
            dependency.dependency for dependency in getattr(route, "dependencies", [])
        }
        if require_internal_secret not in dependencies:
            unprotected.append(path)

    assert unprotected == []
