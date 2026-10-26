# KrckRP — Local, Uncensored, RAG-Driven RPG Platform

KrckRP is a fully local tabletop role-playing engine. A FastAPI orchestrator
wires two persistent memory stores — **SQLite** for exact "hard" game state
and **ChromaDB** for semantic lore retrieval — to a **local KoboldCpp LLM**
serving an OpenAI-compatible API. A React/Tailwind frontend renders the
candlelit-tabletop play surface: narrative chat on the left, live character
state (HP, resources, abilities, inventory, quests, NPCs) on the right.

Everything runs on your hardware. No cloud, no telemetry, no content
filtering.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites & Requirements](#prerequisites--requirements)
3. [Step-by-Step Startup Sequence](#step-by-step-startup-sequence)
4. [Campaign Customization Guide](#campaign-customization-guide)
5. [Gameplay Mechanics & Features](#gameplay-mechanics--features)
6. [Troubleshooting & Maintenance](#troubleshooting--maintenance)

---

## Architecture Overview

```
┌────────────────────────────┐         ┌─────────────────────────────┐
│  React + Tailwind frontend │ ──────► │  FastAPI orchestrator       │
│  (api.ts, GameStateContext,│  HTTP   │  (orchestrator.py, v0.2.0)  │
│   ChatInterface, Sidebar)  │ :8000   │  POST /api/action           │
└────────────────────────────┘         │  GET  /health               │
                                       └──────┬──────────┬───────────┘
                              hard state       │          │       RAG lore
                                       ┌──────▼──────┐ ┌──▼─────────────────┐
                                       │ SQLite      │ │ ChromaDB           │
                                       │ game_state.db│ │ ./chroma_db        │
                                       │ (StateManager)│ (LoreManager, bge-base-en-v1.5)
                                       └─────────────┘ └────────────────────┘
                                                        ┌─────────────────────┐
                                       LLM call        │ KoboldCpp (local)   │
                                       :5001/v1  ─────►│ OpenAI-compatible   │
                                                        │ /chat/completions   │
                                                        └─────────────────────┘
```

| Layer | Technology | Role |
|---|---|---|
| **Backend engine** | Python 3.10+ / FastAPI / Pydantic v2 / httpx | Single `POST /api/action` endpoint that orchestrates state, lore, LLM call, and state-mutation parsing |
| **Hard state** | SQLite (`game_state.db`) via `state_memory/state_manager.py` | Exact facts only: players, HP, level, primary resource pool, location, NPCs, inventory, abilities, quests, chat history |
| **Lore memory** | ChromaDB (`./chroma_db`, collection `world_lore`) via `lore_manager.py` | Chunked, embedded world lore (SentenceTransformer `BAAI/bge-base-en-v1.5`) for semantic RAG retrieval; device auto-selection: CUDA → DirectML → CPU |
| **LLM** | KoboldCpp (local, GGUF model) | OpenAI-compatible server on `http://localhost:5001/v1`, called through `LLMClient` |
| **Frontend** | React 18 + TypeScript + Tailwind (Vite) | `api.ts` (typed fetch client), `GameStateContext.tsx` (state + submit pipeline), `ChatInterface.tsx` (narrative log), `Sidebar.tsx` (live stat/inventory/quest panels) |

**Per-request pipeline** (each player action):

1. `StateManager.get_player_state(player_id)` — hard state snapshot
2. `LoreManager.search_lore(action, location_id)` — location-scoped RAG (global fallback if the location query is empty)
3. `StateManager.get_recent_chat(player_id)` — sliding window of past turns
4. `build_messages(...)` — assembles system prompt (from `system_prompts.json` + state + lore) → chat history → current action
5. `LLMClient.complete(messages)` — KoboldCpp `/chat/completions`
6. `parse_and_apply_mutations(...)` — extracts `<state_update>` JSON, applies it to SQLite, strips it from the narrative
7. Both turns appended to `ChatHistory`; response returns cleaned narrative + retrieved lore + fresh state

---

## Prerequisites & Requirements

### Software

| Dependency | Version | Notes |
|---|---|---|
| Python | **3.10+** | `fastapi`, `uvicorn`, `httpx`, `pydantic` v2, `chromadb`, `sentence-transformers`, `torch` |
| Node.js | **18+** | For the Vite/React frontend |
| KoboldCpp | latest release | Single executable; must expose the **OpenAI-compatible** endpoint (it does by default on recent builds) |
| GGUF model | any uncensored tabletop-capable model | e.g. 7B–14B Q4/Q5 quant; downloaded separately (Hugging Face, etc.) |

### Hardware recommendation

| Tier | GPU | System RAM | What it runs comfortably |
|---|---|---|---|
| **Recommended** | **16 GB VRAM** — AMD **Radeon RX 7900 XT** (16 GB) or NVIDIA **RTX 4080** (16 GB) | 32 GB | 14B Q5 at **32k context** with KV-cache headroom; embeddings on GPU |
| **Minimum** | 8 GB VRAM — AMD RX 6800 / NVIDIA RTX 3070 | 16 GB | 7B Q4 at **16k context** with `--quant-kv 1` |

Notes:

- The embedding model (`BAAI/bge-base-en-v1.5`, ~110 M params) is small and will
  happily run on CPU or DirectML if the GPU is fully committed to the LLM —
  `LoreManager` tries CUDA → DirectML → CPU → library default, in that order,
  and logs its choice at startup.
- On AMD cards, use a KoboldCpp build with Vulkan or HIP enabled so flash
  attention and full GPU offload work.

---

## Step-by-Step Startup Sequence

Run the four steps **in order**, each in its own terminal.

### Step 1 — KoboldCpp Setup (port 5001)

Launch KoboldCpp with your GGUF model on **port 5001**, enabling
**Flash Attention** and **Context Shift**:

```bash
# Windows
koboldcpp.exe ^
  -m "C:\models\your-tabletop-model.gguf" ^
  --port 5001 ^
  --context-shift ^
  --flash-attn 2 ^
  --n-gpu-layers 99

# Linux / macOS
./koboldcpp -m /path/to/your-tabletop-model.gguf \
  --port 5001 \
  --context-shift \
  --flash-attn 2 \
  --n-gpu-layers 99
```

| Flag | Purpose |
|---|---|
| `-m <model.gguf>` | Path to your GGUF model |
| `--port 5001` | Must match the orchestrator's `KOBOLD_BASE_URL` (default `http://localhost:5001/v1`) |
| `--context-shift` | Recycles the oldest context tokens so long sessions never hard-stop |
| `--flash-attn 2` | Flash Attention 2 (use `1` for FA1, `0` to disable) |
| `--n-gpu-layers 99` | Offload all transformer layers to the GPU (lower it if VRAM is tight) |

Verify the OpenAI-compatible endpoint is live:

```bash
curl http://localhost:5001/v1/models
```

You should see a JSON listing with your model name.

> Flag names can vary slightly between KoboldCpp builds — run
> `koboldcpp --help` to confirm. The orchestrator only requires that
> `http://localhost:5001/v1/chat/completions` answers OpenAI-style requests.

### Step 2 — Campaign Ingestion (world_builder.py)

Populate **both** stores from your campaign template *before* the first
game session:

```bash
python world_builder.py                        # uses ./world_data.json
python world_builder.py path/to/world_data.json   # or any campaign file
```

What happens (dependency-safe order):

1. **Locations** → SQLite `Locations`
2. **NPCs** → SQLite `NPCs` (references their `location_id`)
3. **Items** → SQLite `Inventory` (starter pack)
4. **Quests** → SQLite `Quests`
5. **Abilities** → SQLite `Abilities` (learned skills + resource costs)
6. **Lore** → chunked (sliding word-window) and embedded into the ChromaDB
   collection `world_lore`

Behavior notes:

- **Idempotent**: every write is an upsert (SQLite `ON CONFLICT` / content-hashed
  ChromaDB ids). Re-running after edits is safe — it refreshes, never duplicates.
- **Fault-tolerant**: a single bad entry is skipped with a warning; a broken
  *structure* (invalid JSON, missing top-level object) fails loudly with exit 1.
- First run downloads the embedding model (~440 MB) if not cached — needs
  network access **once** only.

### Step 3 — FastAPI Backend (port 8000)

```bash
uvicorn orchestrator:app --reload --port 8000
```

(Alternatively `python orchestrator.py` uses the built-in `run()` entrypoint.)

Sanity-check with the liveness probe — it reports exactly what the engine will
use, with no LLM call and no heavy model init:

```bash
curl http://localhost:8000/health
# {"status":"ok","kobold_base_url":"http://localhost:5001/v1","model":"local-model",
#  "state_db":"game_state.db","chroma_db":"./chroma_db","chroma_collection":"world_lore",
#  "embedding_model":"BAAI/bge-base-en-v1.5"}
```

All configuration is environment-overridable (useful for non-default setups):

| Env var | Default | Meaning |
|---|---|---|
| `KOBOLD_BASE_URL` | `http://localhost:5001/v1` | KoboldCpp OpenAI-compatible base URL |
| `KOBOLD_API_KEY` | `not-needed` | Bearer token, if you enabled auth |
| `KOBOLD_MODEL` | `local-model` | Model name sent in the request |
| `KOBOLD_TIMEOUT` | `180` (s) | Per-request LLM timeout |
| `STATE_DB_PATH` | `game_state.db` | SQLite path |
| `CHROMA_DB_PATH` | `./chroma_db` | ChromaDB persist path |
| `CHROMA_COLLECTION` | `world_lore` | Vector collection name |
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | SentenceTransformer model |
| `SYSTEM_PROMPTS_PATH` | `system_prompts.json` | GM prompt config file |
| `FALLBACK_TO_GLOBAL_LORE` | `1` | Allow unscoped lore search when location query misses |
| `LLM_TEMPERATURE` | `0.85` | Sampling temperature |
| `RECENT_CHAT_LIMIT` | `15` | Sliding chat window size (turns replayed per prompt) |

### Step 4 — Frontend UI (React)

```bash
cd frontend
npm install
npm run dev
```

Open the Vite dev-server URL it prints (typically `http://localhost:5173`).
The API client in `frontend/src/api.ts` targets `http://localhost:8000` —
if you moved the backend, update `API_BASE_URL` there.

You should now see the candlelit tabletop UI: the **chat/narrative panel**
(left) and the **character sidebar** (right: stats & HP bar, resource pool &
abilities, inventory, quests, NPCs in the area). The default dev player is
`kaelen` (seeded in `game_state.db`). Type an action and play.

---

## Campaign Customization Guide

### Tuning the DM: `system_prompts.json`

The Game Master's persona and mechanics live entirely in **one JSON file** —
no code changes required. Four string keys:

| Key | Controls |
|---|---|
| `base_dm_rules` | GM persona: continuity voice, grounding in hard state + lore, NPC disposition consistency, quest-tracking duty, tone (gritty, uncensored), second-person present tense, pacing (1–3 paragraphs, end on a choice or tension), fourth-wall rule |
| `combat_rules` | Turn-by-turn resolution, damage sizing (typical `hp_change` 1–5 per exchange, fair chance to avoid lethal), level-scaled difficulty, persistent consequences (consumed items), same-turn quest progression |
| `resource_rules` | The universal resource economy: the player's primary pool (`Stamina` / `Mana` / `Focus` / `Tech-Charges` — whatever `resource_name` says in state), ability `resource_cost` enforcement, recovery pacing, clamping to `[0, max]` |
| `formatting_instructions` | The **State Mutation Protocol**: the exact `<state_update>` JSON contract the LLM must emit when hard state changes (keys, omission rules, single-line JSON, nothing after the block) |

**Workflow:** edit the JSON → **restart the backend** (prompts are loaded at
import time, so `--reload` does *not* pick up JSON-only edits) → play.

Example — make the GM deadlier and more generous with loot:

```json
"combat_rules": "Combat and difficulty guidelines:\n- Resolve fights turn by turn...\n- Damage: 3–8 per exchange at level 1...\n- Loot is frequent: dangerous encounters should almost always drop an inventory_add item."
```

> ⚠️ Keep the `<state_update>` protocol in `formatting_instructions` intact
> (or extend it carefully) — the server-side parser in `orchestrator.py`
> depends on that exact contract to drive the UI.

### Building your world: `world_data.json`

Top-level shape (every section is a list of objects; sections may be empty or
omitted):

```jsonc
{
  "world_name": "The Drowned Marches — Oakhaven Campaign",
  "default_player_id": "kaelen",

  "locations": [
    { "location_id": "oakhaven_swamp", "name": "Oakhaven Swamp",
      "description": "A fog-choked marsh of black water and dead cypress..." }
  ],

  "npcs": [
    { "npc_id": "vaelen", "name": "Vaelen",
      "location_id": "oakhaven_swamp", "disposition_score": -3 }
  ],

  "lore": [
    { "text": "The Obsidian Brotherhood keeps its vigil in the black bogs...",
      "lore_type": "faction",                 // faction | character | location | item | event
      "character_id": "vaelen",               // optional — ties lore to an NPC
      "location_id": "oakhaven_swamp",        // optional — scopes RAG retrieval to a place
      "source_id": "lore_brotherhood_01",     // unique per entry (upsert identity)
      "chunk_size": 60, "overlap": 15 }       // optional word-window tuning (defaults 250/50)
  ],

  "quests": [
    { "quest_id": "swamp_mystery", "title": "The Swamp's Mystery",
      "description": "Learn why the Brotherhood's iron bells toll...",
      "status": "active" }                    // active | in_progress | completed | failed
  ],

  "items": [
    { "item_id": "potion_1", "item_name": "Healing Potion", "quantity": 2 }
  ],

  "abilities": [
    { "ability_id": "tideglass_sense",
      "name": "Tideglass Sense — feel the vigil bell's warmth...",
      "resource_cost": 5 }
  ]
}
```

**Field rules (enforced at ingestion):**

| Section | Required | Optional |
|---|---|---|
| `locations` | `location_id`, `name` | `description` |
| `npcs` | `npc_id`, `name`, `location_id` (must reference a location) | `disposition_score` (signed int; negative = hostile) |
| `lore` | `text`, `source_id` | `lore_type`, `location_id`, `character_id`, `chunk_size`, `overlap` |
| `quests` | `quest_id`, `title`, `description`, `status` | — |
| `items` | `item_id`, `item_name` | `quantity` (positive int, default 1) |
| `abilities` | `ability_id`, `name` | `resource_cost` (non-negative int, default 0) |

**Tips for good RAG lore:**

- Write lore in **self-contained paragraphs** (150–400 words). Chunks are
  sliding word-windows, so a 60-word window over a 600-word backstory yields
  overlapping, re-embeddable pieces — each should still read sensibly.
- Always set `location_id` on lore that belongs to a place; the orchestrator
  scopes retrieval to the player's current location first, so well-tagged lore
  surfaces exactly when you're standing in that swamp.
- `lore_type` + `character_id` become ChromaDB metadata and can be filtered on
  in advanced queries.
- After editing the file, **re-run `python world_builder.py`** — upserts make
  this safe. (Deleting an entry from the JSON does *not* remove it from the
  databases; see [Database Resets](#database-resets) for a full wipe.)

---

## Gameplay Mechanics & Features

### Sliding chat context (automatic)

- Every player action triggers a full prompt rebuild: **system message**
  (GM rules + authoritative current state + retrieved lore) → the last
  **`RECENT_CHAT_LIMIT` turns** (default **15**, oldest-first) from the
  `ChatHistory` table → the **current action** as the final user message.
- The full history is persisted in SQLite forever; the sliding window only
  controls what the LLM *sees* this turn, keeping prompts bounded and cost
  predictable while the session runs for hours.
- The frontend keeps its own optimistic chat log: your action appears
  immediately, the DM's narrative appends on success; on failure your message
  stays and an error banner is shown — no fake replies are ever injected.

### RAG lore retrieval (automatic)

Per action, the orchestrator:

1. Reads the player's `current_location_id` from hard state.
2. Runs a semantic query over `world_lore` **filtered to that location's
   metadata**, requesting `n_lore_results` chunks (1–20, default **3**).
3. If the scoped query returns nothing, it **falls back to a global search**
   (disable with `FALLBACK_TO_GLOBAL_LORE=0`) so the GM always has *some*
   world context.
4. The winning chunks (with distance + tags) are injected into the system
   prompt under `## RETRIEVED LORE CONTEXT`, and are also echoed back in
   `retrieved_lore` on every response for auditing.

You never write queries — the player's action text *is* the query.

### State mutations via `<state_update>` tags (automatic)

When an action changes hard state, the GM is instructed (via
`formatting_instructions`) to append **exactly one** machine-readable block at
the very end of the narrative:

```
<state_update>{"hp_change": -2, "resource_change": -15, "ability_add": [{"ability_id": "shadow_step", "name": "Shadow Step", "resource_cost": 20}], "inventory_add": [{"item_id": "torch_1", "item_name": "Torch", "quantity": 1}], "inventory_remove": ["potion_1"], "quest_add": [...], "quest_updates": [{"quest_id": "swamp_mystery", "status": "completed"}]}</state_update>
```

Server-side, `parse_and_apply_mutations()` (tag-anchored, non-greedy regex —
narrative prose that merely *mentions* JSON is never touched) then:

| Key | Effect (SQLite) |
|---|---|
| `hp_change` | Integer delta on player HP (negative = damage, positive = healing) |
| `resource_change` | Integer delta on the primary resource pool, **clamped to `[0, max_resource]`** |
| `ability_add` | `[{"ability_id", "name", "resource_cost"}]` — permanently grants new skills (re-sending an id refreshes it) |
| `inventory_add` | `[{"item_id", "item_name", "quantity"}]` — items gained |
| `inventory_remove` | `["item_id", ...]` — items consumed/lost |
| `quest_add` | New quests granted this turn |
| `quest_updates` | `{"quest_id", "status"}` progression of existing quests (`active` / `in_progress` / `completed` / `failed`) |

Protocol discipline (enforced in the prompt): omit unchanged keys, never pad
with zeros, single-line valid JSON, nothing after the block. Malformed blocks
are logged and ignored — the narrative still ships, the state simply doesn't
change.

**The UI updates itself:** every response re-reads fresh state from SQLite and
returns it as `current_state`; `GameStateContext` swaps it in, and the sidebar
re-renders instantly — HP bar, resource progress bar, ability list with costs,
inventory grid, quest status (completed quests dim with a checkmark), and
NPCs present at the current location.

---

## Troubleshooting & Maintenance

### Port / connection errors (HTTP 502 / 500)

A **502** from `POST /api/action` means the orchestrator could not reach or
use KoboldCpp. Work down this checklist:

1. **Is KoboldCpp actually running?**
   `curl http://localhost:5001/v1/models` — must return JSON. If it
   connection-refuses, the process died or bound a different port.
2. **Port mismatch?** The backend defaults to `http://localhost:5001/v1`.
   Check `curl http://localhost:8000/health` — it prints the exact
   `kobold_base_url` in use. If you launched KoboldCpp on another port, either
   change the port or set `KOBOLD_BASE_URL`.
3. **Model not loaded / still loading.** Long GGUF loads and the first
   (prompt-cache-warming) generation can be slow. The default LLM timeout is
   **180 s** (`KOBOLD_TIMEOUT`) — raise it for big models, or watch the
   KoboldCpp console for OOM/abort messages.
4. **VRAM exhaustion.** If the LLM server OOMs mid-generation, reduce context
   or enable KV-cache quantization (below) and restart it.
5. **Wrong API surface.** Ensure your KoboldCpp build exposes the
   **OpenAI-compatible** endpoint (`/v1/chat/completions`), not just the
   legacy Kobold API.
6. **404 "Player ... not found"** is *not* a connectivity problem — the
   `player_id` you're sending doesn't exist in `game_state.db`. Use the seeded
   id (default `kaelen`) or upsert a player via `StateManager.upsert_player`.

Other common symptoms:

- **Embedding model fails to init** on first backend start: `LoreManager`
  tries `cuda → dml → cpu → default`. If all fail (usually no network for the
  one-time download), download `BAAI/bge-base-en-v1.5` into the Hugging Face
  cache manually, then restart.
- **Ingestion exit 1**: the JSON *structure* is invalid (not an object, or a
  section that isn't a list). A single skipped entry, by contrast, only logs a
  warning — read the console for which one you hit.
- **Lore feels off-topic**: confirm your lore entries carry the right
  `location_id`; scoped retrieval won't surface them otherwise (global fallback
  only kicks in when the scoped query is *empty*, not merely weak).

### Database resets (fresh game)

Both stores are plain files on disk — a reset is delete + re-ingest:

```bash
# 1. Stop the backend (Ctrl-C in its terminal)

# 2. Wipe both memory stores
rm game_state.db
rm -r ./chroma_db
# Windows:  del game_state.db && rmdir /s /q chroma_db

# 3. Re-ingest your campaign (recreates both stores)
python world_builder.py path/to/world_data.json

# 4. Restart the backend (Step 3)
uvicorn orchestrator:app --reload --port 8000
```

Granular resets without a full wipe:

- **Lore only**: delete `./chroma_db` (or drop the `world_lore` collection) and
  re-run `world_builder.py` — SQLite state is untouched.
- **State only**: delete `game_state.db` — but note it also holds the seeded
  player and the chat history; re-seeding the player is required before play.
- **One quest/item/NPC**: remove it with a SQLite query (`sqlite3
  game_state.db "DELETE FROM Quests WHERE quest_id='...';"`) — the upsert model
  has no server-side "remove" endpoint, so direct DB edits are the tool of
  choice.

### VRAM overhead: context window & KV cache

The dominant VRAM consumer is the **KV cache**, which scales with
`context_size × batch × layers`. If you're close to the limit:

| Lever | What to do | Effect |
|---|---|---|
| **KV-cache quantization** | `--quant-kv 1` (q8_0) — or `--quant-kv 2` (q4_0) for ~half of that | Cuts KV-cache VRAM ~4× (q8) / ~8× (q4) with minor quality loss |
| **Context size** | Lower `--context-size` from **32768 → 16384** | Halves KV-cache VRAM; fine for most sessions — the orchestrator's sliding window (15 turns) + 3–10 lore chunks comfortably fits 16k |
| **Layer offload** | Drop `--n-gpu-layers` from 99 to the largest value that fits | Spills layers to CPU (slower, but fits tighter VRAM) |
| **Batch** | Keep `--batch-size` modest (e.g. 512) | Prompt processing buffer, secondary but real |

Rule of thumb for the recommended tier (16 GB, 14B Q5):
**16k context + `--quant-kv 1`** is a comfortable default; **32k context**
wants `--quant-kv 1` *and* headroom check — watch the KoboldCpp console for
the KV-cache size it reports at startup and keep at least ~2 GB free for the
embedding model and OS.

> Note: the 15-turn sliding window means *your* prompt never grows unbounded —
> the big context budget is mostly insurance for long lore blocks and long
> narratives, not for history.

---

## Quick-Reference Cheat Sheet

```bash
# ── Terminal 1: LLM ────────────────────────────────────────────
koboldcpp -m model.gguf --port 5001 --context-shift --flash-attn 2 --n-gpu-layers 99

# ── Terminal 2: ingestion (once per campaign change) ──────────
python world_builder.py path/to/world_data.json

# ── Terminal 3: backend ────────────────────────────────────────
uvicorn orchestrator:app --reload --port 8000

# ── Terminal 4: frontend ───────────────────────────────────────
cd frontend && npm install && npm run dev

# ── Health probes ──────────────────────────────────────────────
curl http://localhost:5001/v1/models     # LLM up?
curl http://localhost:8000/health        # Backend config dump
```

**Key files**

| File | Purpose |
|---|---|
| `orchestrator.py` | FastAPI engine, prompt assembly, `<state_update>` parser |
| `state_memory/state_manager.py` | SQLite hard state + chat history |
| `lore_manager.py` | ChromaDB chunking/embedding/retrieval |
| `world_builder.py` | Campaign ingestion CLI |
| `world_data.json` | Your campaign template (edit + re-ingest) |
| `system_prompts.json` | GM persona, combat, resources, mutation protocol (edit + restart) |
| `frontend/src/api.ts` | Typed API client (`API_BASE_URL`) |
| `frontend/src/GameStateContext.tsx` | Client state + submit pipeline |
| `frontend/src/ChatInterface.tsx` | Narrative chat surface |
| `frontend/src/Sidebar.tsx` | Live stat/inventory/quest panels |


P.S. This project has been discontinued due to me being dissatisfied with the results I got from the bare agentic ai I was using. The continuation of this project can be found in a simmilarly titled repo on this page.
