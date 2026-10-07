from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ..models import Skill

_DEFAULT_DB = Path.home() / ".skillforge" / "skills.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS skills (
    id                TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    description       TEXT,
    tags              TEXT,
    version           TEXT,
    validation_score  REAL NOT NULL DEFAULT 0.0,
    source_trace_id   TEXT,
    data              TEXT NOT NULL,
    created_at        TEXT,
    updated_at        TEXT
);
CREATE INDEX IF NOT EXISTS idx_skills_name  ON skills (name);
CREATE INDEX IF NOT EXISTS idx_skills_score ON skills (validation_score DESC);
"""


class SkillRegistry:
    """
    SQLite-backed store for Skill objects.

    Skills are stored as a JSON blob alongside indexed columns for common
    queries (name search, score filter). Schema migrations are additive only —
    never drop or rename a column in a patch release.
    """

    def __init__(self, db_path: Path | str = _DEFAULT_DB) -> None:
        self._path = Path(db_path).resolve()
        # Prevent path traversal when a caller passes a user-supplied path.
        # Allow anything under home or system temp; reject obvious escapes.
        _allowed_roots = (Path.home().resolve(),)
        if not any(self._path.is_relative_to(r) for r in _allowed_roots):
            import tempfile
            tmp = Path(tempfile.gettempdir()).resolve()
            if not self._path.is_relative_to(tmp):
                raise ValueError(
                    f"db_path '{self._path}' must be under the user home directory."
                )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path))
        conn.row_factory = sqlite3.Row
        return conn

    def save(self, skill: Skill) -> None:
        """Upsert a skill. Replaces the existing record if the id matches."""
        row = (
            skill.id,
            skill.name,
            skill.description,
            json.dumps(skill.tags),
            skill.version,
            skill.validation_score,
            skill.source_trace_id,
            json.dumps(skill.to_dict()),
            skill.created_at,
            skill.updated_at,
        )
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO skills
                   (id, name, description, tags, version, validation_score,
                    source_trace_id, data, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                row,
            )

    def get(self, skill_id: str) -> Skill | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT data FROM skills WHERE id = ?", (skill_id,)
            ).fetchone()
        return Skill.from_dict(json.loads(row["data"])) if row else None

    def search(self, query: str) -> list[Skill]:
        """Case-insensitive substring search over name and description."""
        pattern = f"%{query}%"
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT data FROM skills
                   WHERE name LIKE ? OR description LIKE ?
                   ORDER BY validation_score DESC""",
                (pattern, pattern),
            ).fetchall()
        return [Skill.from_dict(json.loads(r["data"])) for r in rows]

    def list_all(self, *, min_score: float = 0.0) -> list[Skill]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT data FROM skills
                   WHERE validation_score >= ?
                   ORDER BY updated_at DESC""",
                (min_score,),
            ).fetchall()
        return [Skill.from_dict(json.loads(r["data"])) for r in rows]

    def delete(self, skill_id: str) -> bool:
        with self._conn() as conn:
            cursor = conn.execute("DELETE FROM skills WHERE id = ?", (skill_id,))
        return cursor.rowcount > 0

    def count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM skills").fetchone()[0]
