"""Core State Memory module.

SQL-backed store for hard game mechanics (stats, locations, NPCs, inventory)
and multi-turn chat history. Runs alongside the RAG/LoreManager system and is
intentionally independent of it:
every fact that must be exact (HP, levels, quantities, who is where) lives in
SQLite instead of vector search.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from typing import Any, Optional

log = logging.getLogger("state_memory")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS Players (
    id                  TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    hp                  INTEGER NOT NULL,
    max_hp              INTEGER NOT NULL,
    level               INTEGER NOT NULL DEFAULT 1,
    current_location_id TEXT REFERENCES Locations(id),
    resource_name       TEXT    NOT NULL DEFAULT 'Stamina',
    current_resource    INTEGER NOT NULL DEFAULT 100,
    max_resource        INTEGER NOT NULL DEFAULT 100
);

CREATE TABLE IF NOT EXISTS NPCs (
    id                TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    location_id       TEXT REFERENCES Locations(id),
    disposition_score INTEGER NOT NULL DEFAULT 0,
    image_prompt      TEXT
);

CREATE TABLE IF NOT EXISTS Locations (
    id               TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    base_description TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS Inventory (
    item_id   TEXT PRIMARY KEY,
    owner_id  TEXT NOT NULL,
    item_name TEXT NOT NULL,
    quantity  INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS Quests (
    quest_id    TEXT NOT NULL,
    player_id   TEXT NOT NULL,
    title       TEXT NOT NULL,
    description TEXT NOT NULL,
    status      TEXT NOT NULL,
    PRIMARY KEY (quest_id, player_id)
);

CREATE TABLE IF NOT EXISTS Abilities (
    ability_id    TEXT NOT NULL,
    player_id     TEXT NOT NULL REFERENCES Players(id),
    name          TEXT NOT NULL,
    resource_cost INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (ability_id, player_id)
);

CREATE TABLE IF NOT EXISTS ChatHistory (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id TEXT NOT NULL,
    role      TEXT NOT NULL,
    content   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_npcs_location ON NPCs (location_id);
CREATE INDEX IF NOT EXISTS idx_inventory_owner ON Inventory (owner_id);
CREATE INDEX IF NOT EXISTS idx_quests_player ON Quests (player_id);
CREATE INDEX IF NOT EXISTS idx_chat_history_player ON ChatHistory (player_id, id);
CREATE INDEX IF NOT EXISTS idx_abilities_player ON Abilities (player_id);
"""

# Resource columns added in place to Players tables created before the
# universal resource system existed (see StateManager._migrate_resource_columns).
_PLAYER_RESOURCE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("resource_name", "TEXT NOT NULL DEFAULT 'Stamina'"),
    ("current_resource", "INTEGER NOT NULL DEFAULT 100"),
    ("max_resource", "INTEGER NOT NULL DEFAULT 100"),
)


class StateManager:
    """Persistent SQLite store for hard game state, with per-call safe connections."""

    def __init__(self, db_path: str = "game_state.db") -> None:
        """Connect to `db_path` (creating the file if missing) and create all tables if absent.

        Databases created before the universal resource system are migrated in
        place: the three resource columns are added to an existing Players
        table, and the Abilities table is created by the schema script above.
        """
        self._db_path = db_path
        with closing(sqlite3.connect(self._db_path)) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.executescript(_SCHEMA)
            self._migrate_resource_columns(conn)
            conn.commit()

    @staticmethod
    def _migrate_resource_columns(conn: sqlite3.Connection) -> None:
        """Add the resource columns to a Players table that predates them (no-op when present)."""
        existing = {row[1] for row in conn.execute("PRAGMA table_info(Players)")}
        for column, declaration in _PLAYER_RESOURCE_COLUMNS:
            if column not in existing:
                conn.execute(f"ALTER TABLE Players ADD COLUMN {column} {declaration}")

    # ------------------------------------------------------------------ upserts

    def upsert_player(
        self,
        player_id: str,
        name: str,
        hp: int,
        max_hp: int,
        level: int,
        current_location_id: Optional[str] = None,
    ) -> None:
        """Insert a Player row, or update name/hp/max_hp/level/location when the id already exists."""
        with closing(sqlite3.connect(self._db_path)) as conn, conn:
            conn.execute(
                """
                INSERT INTO Players (id, name, hp, max_hp, level, current_location_id)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    hp = excluded.hp,
                    max_hp = excluded.max_hp,
                    level = excluded.level,
                    current_location_id = excluded.current_location_id
                """,
                (player_id, name, hp, max_hp, level, current_location_id),
            )

    def upsert_npc(
        self,
        npc_id: str,
        name: str,
        location_id: Optional[str],
        disposition_score: int = 0,
        image_prompt: Optional[str] = None,
    ) -> None:
        """Insert an NPC row, or update name/location/disposition when the id already exists."""
        with closing(sqlite3.connect(self._db_path)) as conn, conn:
            conn.execute(
                """
                INSERT INTO NPCs (id, name, location_id, disposition_score, image_prompt)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    location_id = excluded.location_id,
                    disposition_score = excluded.disposition_score,
                    image_prompt = excluded.image_prompt
                """,
                (npc_id, name, location_id, disposition_score, image_prompt),
            )

    def upsert_location(
        self,
        location_id: str,
        name: str,
        base_description: str = "",
    ) -> None:
        """Insert a Location row, or update name/base_description when the id already exists."""
        with closing(sqlite3.connect(self._db_path)) as conn, conn:
            conn.execute(
                """
                INSERT INTO Locations (id, name, base_description)
                VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    base_description = excluded.base_description
                """,
                (location_id, name, base_description),
            )

    def upsert_inventory_item(
        self,
        item_id: str,
        owner_id: str,
        item_name: str,
        quantity: int,
    ) -> None:
        """Insert an inventory row, or update owner/name/quantity when the item_id already exists."""
        with closing(sqlite3.connect(self._db_path)) as conn, conn:
            conn.execute(
                """
                INSERT INTO Inventory (item_id, owner_id, item_name, quantity)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(item_id) DO UPDATE SET
                    owner_id = excluded.owner_id,
                    item_name = excluded.item_name,
                    quantity = excluded.quantity
                """,
                (item_id, owner_id, item_name, quantity),
            )

    # ----------------------------------------------------------------- quests

    def upsert_quest(
        self,
        player_id: str,
        quest_id: str,
        title: str,
        description: str,
        status: str,
    ) -> None:
        """Insert a Quest row, or update title/description/status when (quest_id, player_id) already exists."""
        with closing(sqlite3.connect(self._db_path)) as conn, conn:
            conn.execute(
                """
                INSERT INTO Quests (quest_id, player_id, title, description, status)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(quest_id, player_id) DO UPDATE SET
                    title = excluded.title,
                    description = excluded.description,
                    status = excluded.status
                """,
                (quest_id, player_id, title, description, status),
            )

    def update_quest_status(self, player_id: str, quest_id: str, status: str) -> None:
        """Advance a quest's status without touching title/description (no-op if absent)."""
        with closing(sqlite3.connect(self._db_path)) as conn, conn:
            conn.execute(
                "UPDATE Quests SET status = ? WHERE quest_id = ? AND player_id = ?",
                (status, quest_id, player_id),
            )

    # --------------------------------------------------------------- abilities

    def upsert_ability(
        self,
        player_id: str,
        ability_id: str,
        name: str,
        resource_cost: int,
    ) -> None:
        """Insert an Ability row, or update name/resource_cost when (ability_id, player_id) already exists."""
        with closing(sqlite3.connect(self._db_path)) as conn, conn:
            conn.execute(
                """
                INSERT INTO Abilities (ability_id, player_id, name, resource_cost)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(ability_id, player_id) DO UPDATE SET
                    name = excluded.name,
                    resource_cost = excluded.resource_cost
                """,
                (ability_id, player_id, name, resource_cost),
            )

    # ---------------------------------------------------------------- retrieval

    def get_player_state(self, player_id: str) -> Optional[dict[str, Any]]:
        """Return one consolidated dict: player stats (incl. the universal resource pool),
        current location, NPCs there, inventory, abilities, and quests (None if the player is missing)."""
        with closing(sqlite3.connect(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row

            player = conn.execute(
                "SELECT id, name, hp, max_hp, level, current_location_id, "
                "resource_name, current_resource, max_resource "
                "FROM Players WHERE id = ?",
                (player_id,),
            ).fetchone()
            if player is None:
                return None

            location: Optional[dict[str, Any]] = None
            npcs_here: list[dict[str, Any]] = []
            location_id = player["current_location_id"]
            if location_id is not None:
                loc = conn.execute(
                    "SELECT id, name, base_description FROM Locations WHERE id = ?",
                    (location_id,),
                ).fetchone()
                if loc is not None:
                    location = {
                        "id": loc["id"],
                        "name": loc["name"],
                        "base_description": loc["base_description"],
                    }
                npcs_here = [
                    dict(row)
                    for row in conn.execute(
                        "SELECT id, name, disposition_score FROM NPCs "
                        "WHERE location_id = ? ORDER BY name",
                        (location_id,),
                    )
                ]

            inventory = [
                dict(row)
                for row in conn.execute(
                    "SELECT item_id, item_name, quantity FROM Inventory "
                    "WHERE owner_id = ? ORDER BY item_name",
                    (player_id,),
                )
            ]

            quests = [
                dict(row)
                for row in conn.execute(
                    "SELECT quest_id, title, description, status FROM Quests "
                    "WHERE player_id = ? ORDER BY quest_id",
                    (player_id,),
                )
            ]

            abilities = [
                dict(row)
                for row in conn.execute(
                    "SELECT ability_id, name, resource_cost FROM Abilities "
                    "WHERE player_id = ? ORDER BY name",
                    (player_id,),
                )
            ]

        return {
            "player": {
                "id": player["id"],
                "name": player["name"],
                "hp": player["hp"],
                "max_hp": player["max_hp"],
                "level": player["level"],
                "current_location_id": location_id,
                "resource_name": player["resource_name"],
                "current_resource": player["current_resource"],
                "max_resource": player["max_resource"],
            },
            "location": location,
            "npcs_here": npcs_here,
            "inventory": inventory,
            "abilities": abilities,
            "quests": quests,
        }

    # ------------------------------------------------------------ chat history

    def append_chat_turn(self, player_id: str, role: str, content: str) -> None:
        """Insert one chat turn (auto-increment id) for a player and commit."""
        with closing(sqlite3.connect(self._db_path)) as conn, conn:
            conn.execute(
                "INSERT INTO ChatHistory (player_id, role, content) VALUES (?, ?, ?)",
                (player_id, role, content),
            )

    def get_recent_chat(self, player_id: str, limit: int = 15) -> list[dict[str, str]]:
        """Return the most recent `limit` turns, oldest-first, in LLM prompt format.

        Each entry is a ready-to-send message dict: {"role": ..., "content": ...}.
        """
        with closing(sqlite3.connect(self._db_path)) as conn:
            rows = conn.execute(
                "SELECT role, content FROM ChatHistory "
                "WHERE player_id = ? ORDER BY id DESC LIMIT ?",
                (player_id, limit),
            ).fetchall()
        return [{"role": row[0], "content": row[1]} for row in reversed(rows)]

    def get_chat_history(self, player_id: str, limit: int = 20) -> list[dict[str, str]]:
        """Return the player's last ``limit`` chat turns, oldest-first (chronological).

        Each entry is ``{"role": ..., "content": ...}`` -- the same shape
        :meth:`get_recent_chat` produces for the LLM prompt window, but
        intended for the frontend chat log (``GET /api/history/{player_id}``)
        so a page refresh can restore the conversation.
        """
        with closing(sqlite3.connect(self._db_path)) as conn:
            rows = conn.execute(
                "SELECT role, content FROM ChatHistory "
                "WHERE player_id = ? ORDER BY id DESC LIMIT ?",
                (player_id, limit),
            ).fetchall()
        return [{"role": row[0], "content": row[1]} for row in reversed(rows)]

    def update_chat_turn(self, message_id: int, new_content: str) -> None:
        """Update the content of a specific chat turn."""
        with closing(sqlite3.connect(self._db_path)) as conn, conn:
            conn.execute(
                "UPDATE ChatHistory SET content = ? WHERE id = ?",
                (new_content, message_id),
            )

    def delete_chat_turns_after(self, message_id: int) -> None:
        """Delete all chat turns that come after the given message ID."""
        with closing(sqlite3.connect(self._db_path)) as conn, conn:
            conn.execute(
                "DELETE FROM ChatHistory WHERE id > ?",
                (message_id,),
            )

    # --------------------------------------------------------- state mutations

    def update_player_hp(self, player_id: str, hp_change: int) -> None:
        """Apply a relative HP delta atomically, clamped to [0, max_hp].

        A single UPDATE (MAX(0, MIN(max_hp, hp + ?))) — no read-modify-write
        race. No-op if the player id does not exist.
        """
        with closing(sqlite3.connect(self._db_path)) as conn, conn:
            conn.execute(
                "UPDATE Players SET hp = MAX(0, MIN(max_hp, hp + ?)) WHERE id = ?",
                (hp_change, player_id),
            )

    def update_player_resource(self, player_id: str, change: int) -> None:
        """Apply a relative resource delta atomically, clamped to [0, max_resource].

        Mirrors :meth:`update_player_hp`: a single UPDATE
        (MAX(0, MIN(max_resource, current_resource + ?))) — no read-modify-write
        race. No-op if the player id does not exist.
        """
        with closing(sqlite3.connect(self._db_path)) as conn, conn:
            conn.execute(
                "UPDATE Players SET current_resource = MAX(0, MIN(max_resource, current_resource + ?)) "
                "WHERE id = ?",
                (change, player_id),
            )

    def update_player_location(self, player_id: str, location_id: str) -> None:
        """Move the player: set Players.current_location_id to ``location_id``.

        The UPDATE is guarded by an EXISTS check on the Locations table, so a
        hallucinated (unknown) location id is a safe no-op with a warning log
        instead of a FOREIGN KEY violation. No-op if the player id does not
        exist.
        """
        with closing(sqlite3.connect(self._db_path)) as conn, conn:
            cursor = conn.execute(
                "UPDATE Players SET current_location_id = ? "
                "WHERE id = ? AND EXISTS (SELECT 1 FROM Locations WHERE id = ?)",
                (location_id, player_id, location_id),
            )
            if cursor.rowcount == 0:
                log.warning(
                    "location change not applied: unknown player %r or location %r",
                    player_id, location_id,
                )

    def remove_inventory_item(self, item_id: str, owner_id: str) -> None:
        """Delete an inventory entry for the given owner (no-op if absent)."""
        with closing(sqlite3.connect(self._db_path)) as conn, conn:
            conn.execute(
                "DELETE FROM Inventory WHERE item_id = ? AND owner_id = ?",
                (item_id, owner_id),
            )
