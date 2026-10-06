"""Shared fake SQLAlchemy AsyncSession plumbing for service tests.

Each service test monkeypatches the module's `get_session` with a factory returning a
`FakeSession` subclass that overrides `handle()` with its own domain logic; this module
owns only the boilerplate (async context manager, commit/rollback/add, statement log).
"""

import re
from types import SimpleNamespace

_MULTIROW_KEY = re.compile(r"^(.+)_m(\d+)$")


def stmt_kind(stmt) -> str:
    """'Insert' / 'Update' / 'Select' / 'Delete' (dialect inserts report 'Insert' too)."""
    return type(stmt).__name__


def stmt_params(stmt) -> dict:
    """A fresh copy of the statement's bound parameters, safe to mutate."""
    return dict(stmt.compile().params)


def multirow_values(stmt) -> list[dict]:
    """An INSERT's rows: multi-row inserts compile to suffixed params (`col_m0`, `col_m1`, ...)."""
    params = stmt_params(stmt)
    rows: dict[int, dict] = {}
    for key, val in params.items():
        match = _MULTIROW_KEY.match(key)
        if match:
            rows.setdefault(int(match.group(2)), {})[match.group(1)] = val
    if 0 not in rows:  # single-row insert: plain column-named params
        return [params]
    return [rows[i] for i in sorted(rows)]


class FakeSession:
    """AsyncSession stand-in: records statements and delegates each one to `handle()`.

    `executed` holds every executed statement in order; `add()`ed ORM objects go to
    `added` (they aren't statements). Subclasses override `handle()` for results.
    """

    def __init__(self) -> None:
        self.executed: list = []
        self.added: list = []
        self.commits = 0
        self.rollbacks = 0

    async def execute(self, stmt):
        self.executed.append(stmt)
        return await self.handle(stmt)

    async def handle(self, stmt):
        raise AssertionError(f"Unexpected statement type: {stmt_kind(stmt)}")

    def add(self, obj) -> None:
        self.added.append(obj)

    async def commit(self) -> None:
        self.commits += 1

    async def rollback(self) -> None:
        self.rollbacks += 1

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def row_result(row):
    """A result whose `.first()` and `.scalars().first()` both return `row` (dicts become objects)."""
    if isinstance(row, dict):
        row = SimpleNamespace(**row)
    return SimpleNamespace(first=lambda: row, scalars=lambda: SimpleNamespace(first=lambda: row))


class ScriptedSession(FakeSession):
    """Returns the scripted `results` in call order, one per statement; `None` once exhausted."""

    def __init__(self, results=None) -> None:
        super().__init__()
        self.results = list(results or [])

    async def handle(self, stmt):
        return row_result(self.results.pop(0) if self.results else None)
