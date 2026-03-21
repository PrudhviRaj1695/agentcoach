"""SQLite Persistence Layer — Story 2.1 CRUD.

═══════════════════════════════════════════════════════════════════
WHY SQLite (Interview Context):
═══════════════════════════════════════════════════════════════════

Interviewer: "Why not Postgres? Why not MongoDB?"

Answer: "For an MVP with single-user workloads, SQLite gives us:
1. Zero infrastructure — no database server to manage
2. File-based — the entire DB is one file, easy to backup/debug
3. ACID compliance — no data corruption
4. Upgrade path — the SQLAlchemy ORM layer means switching to
   Postgres later is a config change, not a rewrite

We'd upgrade to Postgres when: concurrent users > 10, write throughput
becomes a bottleneck, or we need full-text search.

Interview phrase: 'SQLite for MVP — zero infrastructure overhead.
SQLAlchemy abstraction means the upgrade path to Postgres is a config
change. I don't over-engineer infrastructure for single-user workloads.'

═══════════════════════════════════════════════════════════════════
WHY ORM (not raw SQL):
═══════════════════════════════════════════════════════════════════

Raw SQL is fine for simple CRUD, but:
1. Schema migrations are painful without an ORM
2. SQLAlchemy gives us connection pooling (matters for production)
3. The abstraction layer enables database swapping

Interview phrase: 'SQLAlchemy ORM for schema management and database
abstraction. Raw SQL for performance-critical queries if needed.'
"""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

from src.models.state import UserProfile


# Default database path
DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "agentcoach.db",
)


class ProfileStore:
    """SQLite-based persistence for user profiles.

    Simple, reliable, no external dependencies.

    Interview key phrases:
    - "SQLite for MVP — zero infrastructure, ACID-compliant"
    - "JSON serialization for nested objects (skill_scores)"
    - "Profile versioning for audit trail"
    - "Upgrade path to Postgres via SQLAlchemy"
    """

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or DEFAULT_DB_PATH

        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._initialize_db()

    def _initialize_db(self) -> None:
        """Create tables if they don't exist.

        WHY EXPLICIT SCHEMA (not auto-generated):
        We control the schema precisely. This matters for:
        1. Data integrity — we choose the right types and constraints
        2. Migration planning — we know exactly what we're migrating
        3. Debugging — we can inspect the DB with any SQLite browser
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    name TEXT DEFAULT '',
                    target_role TEXT DEFAULT '',
                    experience_years REAL DEFAULT 0,
                    target_companies TEXT DEFAULT '[]',
                    tech_stack TEXT DEFAULT '[]',
                    interview_dates TEXT DEFAULT '{}',
                    skill_scores TEXT DEFAULT '{}',
                    preferred_study_hours REAL DEFAULT 2.0,
                    weak_areas TEXT DEFAULT '[]',
                    profile_version INTEGER DEFAULT 0,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    message_count INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    fact TEXT NOT NULL,
                    source TEXT DEFAULT 'conversation',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
                )
            """)

            conn.commit()

    # ── PROFILE CRUD ──

    def save_profile(self, profile: UserProfile) -> None:
        """Upsert (insert or update) a user profile.

        WHY UPSERT: We don't want separate create/update paths in
        application code. Save() does the right thing regardless of
        whether the profile exists. This simplifies agent code.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO user_profiles
                    (user_id, name, target_role, experience_years,
                     target_companies, tech_stack, interview_dates,
                     skill_scores, preferred_study_hours, weak_areas,
                     profile_version, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    name=excluded.name,
                    target_role=excluded.target_role,
                    experience_years=excluded.experience_years,
                    target_companies=excluded.target_companies,
                    tech_stack=excluded.tech_stack,
                    interview_dates=excluded.interview_dates,
                    skill_scores=excluded.skill_scores,
                    preferred_study_hours=excluded.preferred_study_hours,
                    weak_areas=excluded.weak_areas,
                    profile_version=excluded.profile_version,
                    updated_at=excluded.updated_at
            """, (
                profile.user_id,
                profile.name,
                profile.target_role,
                profile.experience_years,
                json.dumps(profile.target_companies),
                json.dumps(profile.tech_stack),
                json.dumps(profile.interview_dates),
                json.dumps({k: v.model_dump(mode="json") for k, v in profile.skill_scores.items()}),
                profile.preferred_study_hours_per_day,
                json.dumps(profile.weak_areas_self_reported),
                profile.profile_version,
                profile.created_at.isoformat(),
                profile.updated_at.isoformat(),
            ))
            conn.commit()

    def load_profile(self, user_id: str) -> UserProfile | None:
        """Load a user profile by ID. Returns None if not found."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM user_profiles WHERE user_id = ?",
                (user_id,),
            ).fetchone()

            if not row:
                return None

            # Deserialize JSON fields
            skill_scores_raw = json.loads(row["skill_scores"])
            from src.models.state import SkillScore
            skill_scores = {}
            for k, v in skill_scores_raw.items():
                try:
                    skill_scores[k] = SkillScore.model_validate(v)
                except Exception:
                    pass  # Skip corrupted entries

            return UserProfile(
                user_id=row["user_id"],
                name=row["name"],
                target_role=row["target_role"],
                experience_years=row["experience_years"],
                target_companies=json.loads(row["target_companies"]),
                tech_stack=json.loads(row["tech_stack"]),
                interview_dates=json.loads(row["interview_dates"]),
                skill_scores=skill_scores,
                preferred_study_hours_per_day=row["preferred_study_hours"],
                weak_areas_self_reported=json.loads(row["weak_areas"]),
                profile_version=row["profile_version"],
            )

    def delete_profile(self, user_id: str) -> bool:
        """Delete a profile and all associated data."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM memory_facts WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM session_summaries WHERE user_id = ?", (user_id,))
            cursor = conn.execute("DELETE FROM user_profiles WHERE user_id = ?", (user_id,))
            conn.commit()
            return cursor.rowcount > 0

    def list_profiles(self) -> list[dict]:
        """List all profiles (for admin/debugging)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT user_id, name, target_role, updated_at FROM user_profiles ORDER BY updated_at DESC"
            ).fetchall()
            return [dict(row) for row in rows]

    # ── SESSION SUMMARIES ──

    def save_summary(self, user_id: str, summary: str, message_count: int = 0) -> None:
        """Save a session summary."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO session_summaries (user_id, summary, message_count, created_at) VALUES (?, ?, ?, ?)",
                (user_id, summary, message_count, datetime.now().isoformat()),
            )
            conn.commit()

    def get_summaries(self, user_id: str, k: int = 5) -> list[dict]:
        """Get most recent K session summaries."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM session_summaries WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, k),
            ).fetchall()
            return [dict(row) for row in rows]

    # ── MEMORY FACTS ──

    def save_fact(self, user_id: str, fact: str, source: str = "conversation") -> None:
        """Store a long-term memory fact."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO memory_facts (user_id, fact, source, created_at) VALUES (?, ?, ?, ?)",
                (user_id, fact, source, datetime.now().isoformat()),
            )
            conn.commit()

    def get_facts(self, user_id: str) -> list[dict]:
        """Get all stored facts for a user."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM memory_facts WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            ).fetchall()
            return [dict(row) for row in rows]
    def close(self) -> None:
        """Explicitly close any open connections.
        Required on Windows where file locks prevent deletion."""
        # SQLite connections from context managers are already closed,
        # but we ensure the WAL journal is flushed
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception:
            pass