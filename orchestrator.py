"""FastAPI Orchestrator Engine (v0.2.0).

Connects three subsystems into a single ``POST /api/action`` endpoint:

    1. ``StateManager``  -- SQLite "hard" state (HP, level, location, NPCs, inventory,
                            quests) + multi-turn chat history
    2. ``LoreManager``   -- ChromaDB lore RAG (semantic search over world lore)
    3. ``KoboldCpp``     -- local LLM behind an OpenAI-compatible HTTP API
                            (default ``http://localhost:5001/v1``)

Pipeline per request
--------------------
    PlayerActionRequest
        -> StateManager.get_player_state(player_id)           [hard state]
        -> LoreManager.search_lore(action, location_id)       [RAG context]
        -> StateManager.get_recent_chat(player_id)            [sliding chat window]
        -> build_messages(state, lore, action, chat_history)  [prompt assembly]
        -> LLMClient.complete(messages)                       [KoboldCpp /chat/completions]
        -> parse_and_apply_mutations(raw, player, state_mgr)  [apply <state_update>, clean text]
        -> append_chat_turn("user") + append_chat_turn("assistant")
        -> PlayerActionResponse{cleaned narrative, retrieved_lore, fresh state}

Design notes
------------
* The heavy managers (``LoreManager`` pulls in torch/sentence-transformers,
  ``StateManager`` pulls in SQLite) are imported **lazily** inside their factory
  functions. Importing this module therefore stays cheap and never loads the
  embedding model, which keeps ``test_orchestrator.py`` runnable without VRAM.
* The LLM call is isolated behind :class:`LLMClient` so tests can inject an
  ``httpx.MockTransport`` and exercise the *real* request-building code path
  without a live KoboldCpp server.
* The prompt builder (:func:`build_messages`) and the state-mutation parser
  (:func:`parse_and_apply_mutations`) are pure functions of their inputs and
  can be unit-tested in isolation.
* The GM system prompt is **external**: loaded from ``system_prompts.json``
  (``base_dm_rules`` + ``combat_rules`` + ``resource_rules`` +
  ``formatting_instructions``) into the module-level ``SYSTEM_PROMPTS`` dict
  at import time, so the persona, combat handling, the skills/resource
  economy, and the ``<state_update>`` protocol can be tuned without code
  changes.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional, TYPE_CHECKING

import httpx
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

if TYPE_CHECKING:  # type-checker only; these are imported lazily at runtime
    from lore_manager import LoreManager
    from state_memory.state_manager import StateManager

log = logging.getLogger("orchestrator")

# --------------------------------------------------------------------------- #
# Configuration (overridable via environment)
# --------------------------------------------------------------------------- #
KOBOLD_BASE_URL: str = os.environ.get("KOBOLD_BASE_URL", "http://localhost:5001/v1")
KOBOLD_API_KEY: str = os.environ.get("KOBOLD_API_KEY", "not-needed")
KOBOLD_MODEL: str = os.environ.get("KOBOLD_MODEL", "local-model")
KOBOLD_TIMEOUT: float = float(os.environ.get("KOBOLD_TIMEOUT", "180"))

STATE_DB_PATH: str = os.environ.get("STATE_DB_PATH", "game_state.db")
CHROMA_DB_PATH: str = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
CHROMA_COLLECTION: str = os.environ.get("CHROMA_COLLECTION", "world_lore")
EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

# External GM prompt configuration (single JSON object with the three prompt
# sections below). Relative paths resolve against the current working directory.
SYSTEM_PROMPTS_PATH: str = os.environ.get("SYSTEM_PROMPTS_PATH", "system_prompts.json")

# If the location-scoped RAG query returns nothing, fall back to a global query
# so the LLM still has *some* world context. Set to False to force strict scope.
FALLBACK_TO_GLOBAL_LORE: bool = os.environ.get("FALLBACK_TO_GLOBAL_LORE", "1") not in (
    "0",
    "false",
    "False",
)

LLM_TEMPERATURE: float = float(os.environ.get("LLM_TEMPERATURE", "0.85"))

# Sliding window: how many past chat turns are replayed into each prompt.
RECENT_CHAT_LIMIT: int = int(os.environ.get("RECENT_CHAT_LIMIT", "15"))

# --------------------------------------------------------------------------- #
# Pydantic request / response models
# --------------------------------------------------------------------------- #
class PlayerActionRequest(BaseModel):
    """A single player action to be narrated by the GM."""

    player_id: str = Field(..., min_length=1, description="Unique id of the acting player.")
    action_text: str = Field(
        ...,
        min_length=1,
        description="The player's intended action in prose, e.g. 'I draw my sword and lunge at the guard.'",
    )
    n_lore_results: int = Field(
        3,
        ge=1,
        le=20,
        description="How many RAG lore chunks to retrieve and ground the narration in.",
    )


class PlayerActionResponse(BaseModel):
    """Narrative plus the hard state and lore that produced it (for debugging/audit)."""

    narrative_response: str = Field(
        ..., description="The GM's narrative prose (cleaned — any <state_update> block removed)."
    )
    retrieved_lore: list[dict[str, Any]] = Field(
        ..., description="Raw LoreManager.search_lore() results used to ground the scene."
    )
    current_state: dict[str, Any] = Field(
        ..., description="The freshly re-read StateManager state snapshot after mutation."
    )


# --------------------------------------------------------------------------- #
# LLM client (OpenAI-compatible; KoboldCpp)
# --------------------------------------------------------------------------- #
class LLMClient:
    """Thin wrapper over KoboldCpp's OpenAI-compatible ``/chat/completions``.

    ``client`` is injectable for testing: pass an ``httpx.Client`` built on top
    of ``httpx.MockTransport`` to run the *real* :meth:`complete` logic against
    a canned response.
    """

    def __init__(
        self,
        base_url: str = KOBOLD_BASE_URL,
        api_key: str = KOBOLD_API_KEY,
        model: str = KOBOLD_MODEL,
        timeout: float = KOBOLD_TIMEOUT,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._injected_client = client

    def _get_client(self) -> tuple[httpx.Client, bool]:
        """Return ``(client, owned)``; ``owned`` is True only when we must close it."""
        if self._injected_client is not None:
            return self._injected_client, False
        return httpx.Client(timeout=self.timeout), True

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.85,
        max_tokens: int = 1024,
    ) -> str:
        """Send ``messages`` to ``{base_url}/chat/completions`` and return the text."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        client, close = self._get_client()
        try:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        finally:
            if close:
                client.close()

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"KoboldCpp returned no choices: {data!r}")
        content = (choices[0].get("message") or {}).get("content")
        if content is None:
            raise RuntimeError(f"KoboldCpp returned no message content: {choices[0]!r}")
        return content


# --------------------------------------------------------------------------- #
# Manager singletons (lazily created) + FastAPI dependency providers
# --------------------------------------------------------------------------- #
_state_manager = None
_lore_manager = None
_llm_client: Optional[LLMClient] = None


def get_state_manager():
    """Return the shared :class:`StateManager`, creating it on first use.

    (No return annotation on purpose: this is a FastAPI dependency, and keeping
    the manager type out of the runtime signature lets us lazy-import it.)
    """
    global _state_manager
    if _state_manager is None:
        from state_memory.state_manager import StateManager  # lazy: sqlite3-backed

        log.info("Initialising StateManager at %s", STATE_DB_PATH)
        _state_manager = StateManager(db_path=STATE_DB_PATH)
    return _state_manager


def get_lore_manager():
    """Return the shared :class:`LoreManager`, creating it on first use.

    The first call loads the sentence-transformer embedding model (CPU/GPU), so
    this is intentionally deferred until a real request needs it.
    """
    global _lore_manager
    if _lore_manager is None:
        from lore_manager import LoreManager  # lazy: chromadb + sentence-transformers + torch

        log.info(
            "Initialising LoreManager (chroma=%s, collection=%s, model=%s)",
            CHROMA_DB_PATH, CHROMA_COLLECTION, EMBEDDING_MODEL,
        )
        _lore_manager = LoreManager(
            db_path=CHROMA_DB_PATH,
            collection_name=CHROMA_COLLECTION,
            embedding_model=EMBEDDING_MODEL,
        )
    return _lore_manager


def get_llm_client() -> LLMClient:
    """Return the shared :class:`LLMClient` (one underlying httpx client)."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


# --------------------------------------------------------------------------- #
# External system prompt configuration (system_prompts.json)
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT_KEYS: tuple[str, ...] = (
    "base_dm_rules",
    "combat_rules",
    "resource_rules",
    "formatting_instructions",
)


def _load_system_prompts(path: str = SYSTEM_PROMPTS_PATH) -> dict[str, str]:
    """Load the external GM prompt configuration from a JSON file.

    Expects a single JSON object with the non-empty string keys
    ``base_dm_rules``, ``combat_rules``, ``resource_rules`` and
    ``formatting_instructions``.
    Raises :class:`RuntimeError` with a clear message when the file is
    missing, unparseable, or incomplete, so a broken configuration fails
    loudly at startup instead of mid-request.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        raise RuntimeError(
            f"System prompt config not found at {path!r}. Create a JSON object with "
            "the keys: base_dm_rules, combat_rules, resource_rules, formatting_instructions "
            "(or set SYSTEM_PROMPTS_PATH)."
        ) from None
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"System prompt config at {path!r} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"System prompt config at {path!r} must be a single JSON object.")
    missing = [key for key in SYSTEM_PROMPT_KEYS if not str(data.get(key) or "").strip()]
    if missing:
        raise RuntimeError(f"System prompt config at {path!r} is missing required key(s): {', '.join(missing)}.")
    return {key: str(data[key]).strip() for key in SYSTEM_PROMPT_KEYS}


# Global prompt configuration, loaded once at startup (module import).
SYSTEM_PROMPTS: dict[str, str] = _load_system_prompts()


# --------------------------------------------------------------------------- #
# Prompt assembly (pure, unit-testable)
# ---------------------------------------------------------------------------


def _pick(mapping: Any, *keys: str) -> Any:
    """Return the first non-empty value among ``keys`` in a (maybe) dict."""
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        value = mapping.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _format_inventory(items: Any) -> str:
    if not items:
        return "  (inventory empty)"
    lines = []
    for item in items:
        if isinstance(item, dict):
            name = _pick(item, "item_name", "name", "item")
            qty = item.get("quantity", item.get("qty"))
            line = f"  - {name or 'Unknown item'}"
            if qty is not None:
                line += f" x {qty}"
            lines.append(line)
        else:
            lines.append(f"  - {item}")
    return "\n".join(lines) if lines else "  (inventory empty)"


def _format_quests(quests: Any) -> str:
    if not quests:
        return "  (no active quests)"
    lines = []
    for quest in quests:
        if isinstance(quest, dict):
            # Skip hidden and locked quests to prevent AI from knowing about future quests
            status = quest.get("status")
            if status in ("hidden", "locked"):
                continue
            title = _pick(quest, "title", "name") or _pick(quest, "quest_id", "id")
            line = f"  - {title or 'Unknown quest'}"
            if status is not None:
                line += f" (status: {status})"
            lines.append(line)
        else:
            lines.append(f"  - {quest}")
    return "\n".join(lines) if lines else "  (no active quests)"


def _format_abilities(abilities: Any, resource_label: str = "resource") -> str:
    if not abilities:
        return "  (no skills acquired yet)"
    lines = []
    for ability in abilities:
        if isinstance(ability, dict):
            name = _pick(ability, "name", "ability_name") or _pick(ability, "ability_id", "id")
            cost = ability.get("resource_cost", ability.get("cost"))
            line = f"  - {name or 'Unknown ability'}"
            if cost is not None:
                line += f" ({resource_label} cost: {cost})"
            lines.append(line)
        else:
            lines.append(f"  - {ability}")
    return "\n".join(lines) if lines else "  (no skills acquired yet)"


def _format_npcs(npcs: Any) -> str:
    if not npcs:
        return "  (no NPCs present)"
    lines = []
    for npc in npcs:
        if isinstance(npc, dict):
            name = _pick(npc, "name", "npc_name")
            disp = npc.get("disposition_score", npc.get("disposition"))
            line = f"  - {name or 'Unknown NPC'}"
            if disp is not None:
                line += f" (disposition: {disp})"
            lines.append(line)
        else:
            lines.append(f"  - {npc}")
    return "\n".join(lines) if lines else "  (no NPCs present)"


def format_state_for_prompt(state: dict[str, Any]) -> str:
    """Render the authoritative player state as a readable GM brief.

    Targets the StateManager.get_player_state() shape: player stats live in
    the nested ``player`` dict (``state["player"]["name" / "hp" / "max_hp" /
    "level" / "resource_name" / "current_resource" / "max_resource"]``), while
    location, npcs_here, inventory, abilities, and quests are top-level keys.
    A flat dict (stats at the top level) is still tolerated as a fallback.
    """
    player = state.get("player")
    if not isinstance(player, dict):
        player = state  # tolerate a flat state dict

    name = _pick(player, "name", "player_name") or state.get("player_id")
    hp = _pick(player, "hp", "current_hp")
    max_hp = _pick(player, "max_hp", "maximum_hp", "maxHp")
    level = _pick(player, "level", "player_level")
    resource_name = _pick(player, "resource_name") or "Stamina"
    resource = _pick(player, "current_resource")
    max_resource = _pick(player, "max_resource")
    abilities = state.get("abilities") or player.get("abilities")

    location = state.get("location")
    if not isinstance(location, dict):
        location = {}
    loc_name = _pick(location, "name", "location_name", "title") or _pick(state, "location_name")
    loc_id = (
        _pick(location, "location_id", "id")
        or _pick(player, "current_location_id")
        or _pick(state, "current_location_id", "location_id")
    )

    hp_text = f"{hp}/{max_hp}" if hp is not None and max_hp is not None else (
        f"{hp}" if hp is not None else "unknown"
    )
    resource_text = (
        f"{resource_name} {resource}/{max_resource}"
        if resource is not None and max_resource is not None
        else (f"{resource_name} {resource}" if resource is not None else f"{resource_name} unknown")
    )

    parts = [
        f"  Player: {name or 'Unknown'}",
        f"  HP: {hp_text}",
        f"  Level: {level if level is not None else 'unknown'}",
        f"  Primary Resource: {resource_text}",
        f"  Location: {loc_name or 'Unknown'}" + (f" (id: {loc_id})" if loc_id else ""),
        "",
        "  Co-located NPCs:",
        _format_npcs(state.get("npcs") or state.get("npcs_here") or state.get("co_located_npcs") or state.get("present_npcs")),
        "",
        "  Inventory:",
        _format_inventory(state.get("inventory") or state.get("items") or state.get("inv")),
        "",
        f"  Known Abilities (each spends {resource_name} to use):",
        _format_abilities(abilities, resource_name),
        "",
        "  Quests:",
        _format_quests(state.get("quests")),
    ]
    return "\n".join(parts)


def format_lore_for_prompt(lore: list[dict[str, Any]]) -> str:
    """Render the retrieved RAG lore as numbered, tagged context snippets."""
    if not lore:
        return "  (no lore retrieved for this scene)"
    lines = []
    for idx, entry in enumerate(lore, start=1):
        meta = entry.get("metadata") or {}
        tags = ", ".join(
            f"{k}={v}"
            for k, v in meta.items()
            if k not in ("chunk_index", "total_chunks") and v not in (None, "")
        )
        tag_text = f" [{tags}]" if tags else ""
        distance = entry.get("distance")
        dist_text = f" (distance={distance:.4f})" if isinstance(distance, (int, float)) else ""
        text = (entry.get("text") or "").strip().replace("\n", " ")
        lines.append(f"  {idx}. {text}{dist_text}{tag_text}")
    return "\n".join(lines)


def build_messages(
    state: dict[str, Any],
    lore: list[dict[str, Any]],
    action_text: str,
    chat_history: Optional[list[dict[str, str]]] = None,
) -> list[dict[str, str]]:
    """Assemble the OpenAI-style prompt from state + lore + recent chat + action.

    Message order:
      1. ``system``  — GM instructions loaded from ``system_prompts.json``
         (``base_dm_rules`` + ``combat_rules`` + ``resource_rules`` +
         ``formatting_instructions``, the last carrying the State Mutation
         Protocol incl. ``resource_change``, ``ability_add`` and
         ``quest_updates``), then the authoritative SQLite state (incl. the
         primary resource pool and known abilities) and the retrieved
         ChromaDB lore.
      2. recent chat turns (sliding window, oldest-first) — each validated to a
         ``user``/``assistant`` role with non-empty content.
      3. ``user``    — the player's current action.
    """
    system_parts = [
        SYSTEM_PROMPTS["base_dm_rules"],
        SYSTEM_PROMPTS["combat_rules"],
        SYSTEM_PROMPTS["resource_rules"],
        SYSTEM_PROMPTS["formatting_instructions"],
        "\n## CURRENT WORLD STATE (authoritative -- apply changes only via the State Mutation Protocol)\n"
        + format_state_for_prompt(state),
        "\n## RETRIEVED LORE CONTEXT (ground your narration here)\n"
        + format_lore_for_prompt(lore),
    ]
    system = "\n".join(system_parts).strip()

    user = (
        "The player acts now. Narrate the outcome as the GM.\n"
        f"Player action: {action_text.strip()}"
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    for turn in chat_history or []:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user})
    return messages


# --------------------------------------------------------------------------- #
# State mutation parsing (LLM <state_update> block -> StateManager calls)
# --------------------------------------------------------------------------- #
# Tag-anchored, non-greedy, DOTALL:
#   * <state_update> ... </state_update> anchors the capture to the tags, so
#     narrative prose that merely mentions JSON is never touched.
#   * \s* trims LLM-added whitespace/newlines around the payload.
#   * .*? (non-greedy) stops at the FIRST closing tag, so multiple blocks in one
#     response are captured one-by-one via finditer.
#   * re.DOTALL lets the JSON span multiple lines if the LLM pretty-prints it.
STATE_UPDATE_PATTERN: "re.Pattern[str]" = re.compile(
    r"<state_update>\s*(?P<payload>.*?)\s*</state_update>",
    re.DOTALL,
)


def _loads_json_block(raw: str) -> Optional[dict[str, Any]]:
    """Parse a captured payload as a JSON object; tolerate code fences.

    Returns the dict, or ``None`` (with a warning) when the payload is not a
    valid JSON object.
    """
    text = (raw or "").strip()
    if text.startswith("```"):
        # Some models wrap the JSON in a markdown fence despite the protocol.
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        log.warning("Malformed <state_update> JSON ignored (%s): %r", exc, raw[:200])
        return None
    if not isinstance(data, dict):
        log.warning("<state_update> payload is not a JSON object, ignored: %r", raw[:200])
        return None
    return data


def _apply_mutations(payload: dict[str, Any], player_id: str, state_mgr) -> None:
    """Apply one parsed <state_update> payload through StateManager methods.

    Handles ``hp_change``, ``resource_change`` (relative delta applied via
    :meth:`StateManager.update_player_resource`, clamped to
    ``[0, max_resource]``), ``location_id`` (moves the player via
    :meth:`StateManager.update_player_location`; unknown location ids are a
    safe no-op there), ``ability_add`` (each {"ability_id", "name",
    "resource_cost"} entry grants/refreshes a skill via
    :meth:`StateManager.upsert_ability`), ``inventory_add``,
    ``inventory_remove``, ``quest_updates`` (each {"quest_id", "status"} entry
    advances an existing quest via :meth:`StateManager.update_quest_status`),
    ``quest_add`` (each {"quest_id", "title", "description", "status"}
    entry creates or refreshes a quest via :meth:`StateManager.upsert_quest`),
    and ``npc_add`` (each {"npc_id", "name", "location_id", "image_prompt"}
    entry creates or updates an NPC with the given properties).
    """
    hp_change = payload.get("hp_change")
    if isinstance(hp_change, bool):
        hp_change = None
    if isinstance(hp_change, (int, float)):
        delta = int(hp_change)
        if delta:
            state_mgr.update_player_hp(player_id, delta)
            log.info("hp_change %+d applied to player %s", delta, player_id)

    resource_change = payload.get("resource_change")
    if isinstance(resource_change, bool):
        resource_change = None
    if isinstance(resource_change, (int, float)):
        delta = int(resource_change)
        if delta:
            state_mgr.update_player_resource(player_id, delta)
            log.info("resource_change %+d applied to player %s", delta, player_id)

    new_location_id = payload.get("location_id")
    if isinstance(new_location_id, str) and new_location_id.strip():
        state_mgr.update_player_location(player_id, new_location_id.strip())
        log.info("location_id -> %s for player %s", new_location_id.strip(), player_id)

    ability_adds = payload.get("ability_add")
    if isinstance(ability_adds, list):
        for entry in ability_adds:
            if not isinstance(entry, dict):
                continue
            ability_id = str(entry.get("ability_id") or "").strip()
            ability_name = str(entry.get("name") or ability_id).strip()
            if not ability_id:
                continue
            try:
                resource_cost = max(0, int(entry.get("resource_cost", 0)))
            except (TypeError, ValueError):
                resource_cost = 0
            state_mgr.upsert_ability(player_id, ability_id, ability_name, resource_cost)
            log.info("ability_add %s (%s, cost %d) for player %s", ability_id, ability_name, resource_cost, player_id)

    adds = payload.get("inventory_add")
    if isinstance(adds, list):
        for entry in adds:
            if not isinstance(entry, dict):
                continue
            item_id = str(entry.get("item_id") or "").strip()
            if not item_id:
                continue
            item_name = str(entry.get("item_name") or item_id).strip()
            try:
                quantity = int(entry.get("quantity", 1))
            except (TypeError, ValueError):
                quantity = 1
            if quantity <= 0:
                continue
            state_mgr.upsert_inventory_item(item_id, player_id, item_name, quantity)
            log.info("inventory_add %s x%d for player %s", item_id, quantity, player_id)

    removes = payload.get("inventory_remove")
    if isinstance(removes, list):
        for item_id in removes:
            item_id = str(item_id).strip()
            if not item_id:
                continue
            state_mgr.remove_inventory_item(item_id, player_id)
            log.info("inventory_remove %s for player %s", item_id, player_id)

    quest_updates = payload.get("quest_updates")
    if isinstance(quest_updates, list):
        for entry in quest_updates:
            if not isinstance(entry, dict):
                continue
            quest_id = str(entry.get("quest_id") or "").strip()
            status = str(entry.get("status") or "").strip()
            if not quest_id or not status:
                continue
            state_mgr.update_quest_status(player_id, quest_id, status)
            log.info("quest %s -> %s for player %s", quest_id, status, player_id)

    quest_adds = payload.get("quest_add")
    if isinstance(quest_adds, list):
        for entry in quest_adds:
            if not isinstance(entry, dict):
                continue
            quest_id = str(entry.get("quest_id") or "").strip()
            title = str(entry.get("title") or "").strip()
            if not quest_id or not title:
                log.warning("quest_add entry skipped (quest_id and title required): %r", entry)
                continue
            description = str(entry.get("description") or "").strip()
            status = str(entry.get("status") or "active").strip() or "active"
            state_mgr.upsert_quest(player_id, quest_id, title, description, status)
            log.info("quest_add %s (%s) for player %s", quest_id, status, player_id)

    npc_adds = payload.get("npc_add")
    if isinstance(npc_adds, list):
        for entry in npc_adds:
            if not isinstance(entry, dict):
                continue
            npc_id = str(entry.get("npc_id") or "").strip()
            name = str(entry.get("name") or "").strip()
            location_id = str(entry.get("location_id") or "").strip()
            image_prompt = str(entry.get("image_prompt") or "").strip()
            if not npc_id or not name or not location_id:
                log.warning("npc_add entry skipped (npc_id, name and location_id required): %r", entry)
                continue
            state_mgr.upsert_npc(npc_id, name, location_id, 0, image_prompt)
            log.info("npc_add %s (%s) at %s for player %s", npc_id, name, location_id, player_id)


def parse_and_apply_mutations(
    narrative_response: str,
    player_id: str,
    state_mgr: "StateManager",
) -> str:
    """Apply any ``<state_update>`` block(s) and return the cleaned narrative.

    1. Locates every ``<state_update>...</state_update>`` block via regex.
    2. Parses each payload as JSON and applies ``hp_change`` /
       ``resource_change`` / ``location_id`` / ``ability_add`` / ``inventory_add`` /
       ``inventory_remove`` / ``quest_updates`` / ``quest_add`` through
       StateManager (``resource_change`` calls ``update_player_resource``;
       ``ability_add`` entries call ``upsert_ability``; ``quest_updates``
       entries call ``update_quest_status``; ``quest_add`` entries call
       ``upsert_quest``).
    3. Returns the narrative with all blocks removed and trimmed.

    Malformed JSON blocks are logged and skipped, but the block is still
    stripped from the user-visible narrative. A response with no block is
    returned unchanged (aside from trimming).
    """
    if not narrative_response:
        return ""
    matches = list(STATE_UPDATE_PATTERN.finditer(narrative_response))
    for match in matches:
        payload = _loads_json_block(match.group("payload"))
        if payload is None:
            continue
        _apply_mutations(payload, player_id, state_mgr)
    return STATE_UPDATE_PATTERN.sub("", narrative_response).strip()


# --------------------------------------------------------------------------- #
# FastAPI app + endpoint
# --------------------------------------------------------------------------- #
def create_app() -> FastAPI:
    """Build the FastAPI application (separated so tests can import pieces)."""
    app = FastAPI(
        title="KrckRP Orchestrator",
        description="FastAPI engine wiring SQLite hard state + ChromaDB lore RAG to a local KoboldCpp LLM.",
        version="0.2.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allows all origins (including http://localhost:5173)
        allow_credentials=True,
        allow_methods=["*"],  # Allows POST, GET, OPTIONS, etc.
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        """Liveness probe (no LLM call, no heavy init)."""
        return {
            "status": "ok",
            "kobold_base_url": KOBOLD_BASE_URL,
            "model": KOBOLD_MODEL,
            "state_db": STATE_DB_PATH,
            "chroma_db": CHROMA_DB_PATH,
            "chroma_collection": CHROMA_COLLECTION,
            "embedding_model": EMBEDDING_MODEL,
        }

    @app.get("/api/history/{player_id}")
    def get_chat_history(
        player_id: str,
        state_mgr=Depends(get_state_manager),  # -> StateManager
    ) -> dict[str, Any]:
        """Return the player's recent chat history (last 20 turns, oldest-first).

        The frontend calls this on mount / player switch so the log survives
        a page refresh. 404 when the player id is unknown.
        """
        if state_mgr.get_player_state(player_id) is None:
            raise HTTPException(
                status_code=404,
                detail=f"Player '{player_id}' not found in state. "
                "Upsert the player first (StateManager.upsert_player).",
            )
        messages = state_mgr.get_chat_history(player_id, limit=20)
        return {
            "player_id": player_id,
            "messages": [
                {"id": index, "role": msg["role"], "content": msg["content"]}
                for index, msg in enumerate(messages, start=1)
            ],
        }

    @app.put("/api/chat/{message_id}")
    def update_chat_turn(
        message_id: int,
        new_content: str,
        state_mgr=Depends(get_state_manager),  # -> StateManager
    ) -> dict[str, Any]:
        """Update a chat turn and delete all subsequent turns to branch the timeline.

        This allows manual editing of past messages without leaving outdated context in the prompt.
        """
        try:
            state_mgr.update_chat_turn(message_id, new_content)
            state_mgr.delete_chat_turns_after(message_id)
            return {"status": "success", "message_id": message_id}
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to update chat turn: {exc}",
            )

    @app.post("/api/action", response_model=PlayerActionResponse)
    def player_action(
        req: PlayerActionRequest,
        state_mgr=Depends(get_state_manager),  # -> StateManager
        lore_mgr=Depends(get_lore_manager),  # -> LoreManager
        llm: LLMClient = Depends(get_llm_client),
    ) -> PlayerActionResponse:
        # 1) Hard state -----------------------------------------------------
        state = state_mgr.get_player_state(req.player_id)
        if state is None:
            raise HTTPException(
                status_code=404,
                detail=f"Player '{req.player_id}' not found in state. "
                "Upsert the player first (StateManager.upsert_player).",
            )

        # 2) RAG lore -------------------------------------------------------
        location_id = _pick(state, "current_location_id", "location_id")
        location = state.get("location")
        if not location_id and isinstance(location, dict):
            location_id = _pick(location, "location_id", "id")

        metadata_filter = {"location_id": location_id} if location_id else None
        lore: list[dict[str, Any]] = []
        try:
            lore = lore_mgr.search_lore(
                req.action_text,
                n_results=req.n_lore_results,
                metadata_filter=metadata_filter,
            )
            if not lore and metadata_filter and FALLBACK_TO_GLOBAL_LORE:
                log.info("Location-scoped lore empty; falling back to global query.")
                lore = lore_mgr.search_lore(
                    req.action_text, n_results=req.n_lore_results, metadata_filter=None
                )
        except Exception as exc:  # noqa: BLE001 - degrade gracefully, still narrate
            log.warning("Lore retrieval failed; narrating with state only: %s", exc)
            lore = []

        # 3) Chat history (sliding window) ----------------------------------
        chat_history = state_mgr.get_recent_chat(req.player_id, limit=RECENT_CHAT_LIMIT)

        # Event override system - Check for active events and modify NPC grouping if needed
        active_event = state.get("active_event")
        if active_event == "meet_and_greet":
            # Override NPCs in the player's location with specific characters for meet and greet event
            # Define hardcoded list of NPC dictionaries to use during this event
            meet_and_greet_npcs = [
                {
                    "npc_id": "npc_001",
                    "name": "Professor McGonagall",
                    "disposition_score": 5,
                    "location_id": location_id,
                    "image_prompt": "A strict but kind witch in her Hogwarts robes, with a stern look"
                },
                {
                    "npc_id": "npc_002",
                    "name": "Dumbledore",
                    "disposition_score": 7,
                    "location_id": location_id,
                    "image_prompt": "An elderly wizard with long white beard, kind eyes, and a knowing smile"
                }
            ]
            # Update the state to use these specific NPCs during meet_and_greet
            if "npcs" in state:
                state["npcs"] = meet_and_greet_npcs
            elif "npcs_here" in state:
                state["npcs_here"] = meet_and_greet_npcs
            else:
                # If no existing NPC structure, create it
                if not isinstance(state.get("present_npcs"), list):
                    state["present_npcs"] = []
                state["present_npcs"] = meet_and_greet_npcs

        # 4) Prompt + LLM ---------------------------------------------------
        messages = build_messages(
            state,
            lore,
            req.action_text,
            chat_history=chat_history,
        )
        try:
            raw_narrative = llm.complete(messages, temperature=LLM_TEMPERATURE)
        except httpx.HTTPError as exc:
            log.error("KoboldCpp request failed: %s", exc)
            raise HTTPException(
                status_code=502,
                detail=f"LLM request to KoboldCpp ({KOBOLD_BASE_URL}) failed: {exc}",
            ) from exc
        except Exception as exc:  # noqa: BLE001 - non-transport LLM error
            log.error("LLM call failed: %s", exc)
            raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}") from exc

        # 5) Apply state mutations + clean the narrative -------------------
        narrative = parse_and_apply_mutations(raw_narrative, req.player_id, state_mgr)

        # 6) Persist the turn: the user's action, then the cleaned GM reply -
        state_mgr.append_chat_turn(req.player_id, "user", req.action_text)
        state_mgr.append_chat_turn(req.player_id, "assistant", narrative)

        # 7) Respond with the freshly updated state -------------------------
        fresh_state = state_mgr.get_player_state(req.player_id)
        return PlayerActionResponse(
            narrative_response=narrative,
            retrieved_lore=lore,
            current_state=fresh_state if fresh_state is not None else state,
        )

    return app


app = create_app()


# --------------------------------------------------------------------------- #
# Convenience server entrypoint
# --------------------------------------------------------------------------- #
def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    log.info("Starting KrckRP orchestrator on http://%s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()