# Astar Island — Viking Civilisation Prediction

## Overview

Observe a black-box Norse civilisation simulator through a limited viewport and predict the final world state.

The simulator runs a procedurally generated Norse world for **50 years** — settlements grow, factions clash, trade routes form, alliances shift, forests reclaim ruins, and harsh winters reshape entire civilisations.

**Goal:** Predict the **probability distribution** of terrain types across the entire map for each of 5 seeds.

- **Task type:** Observation + probabilistic prediction
- **Platform:** app.ainm.no
- **API base:** `https://api.ainm.no/astar-island/`

---

## Hard Constraints ← KNOW THESE BY HEART

| Parameter | Value | Notes |
|---|---|---|
| **TOTAL_QUERIES** | **50** | Shared across ALL 5 seeds — cannot be undone |
| **QUERIES_PER_SEED** | ~10 | 50 / 5 — use strategically |
| **MAP_SIZE** | **40×40 = 1,600 cells** | Confirmed from API |
| **VIEWPORT_MAX** | **15×15 = 225 cells** | Min width/height is 5 |
| **TILES_TO_COVER_MAP** | 9 | ceil(40/15)=3 → 3×3 grid |
| **SPARE_QUERIES** | 5 | After 9 tiles × 5 seeds = 45 |
| **NUM_SEEDS** | 5 | seed_index 0–4 |
| **NUM_TERRAIN_CLASSES** | 6 | classes 0–5 (prediction layer) |
| **SIMULATION_YEARS** | 50 | Years per sim run |
| **RATE_LIMIT_SIMULATE** | 5 req/sec | 429 if exceeded |
| **RATE_LIMIT_SUBMIT** | 2 req/sec | 429 if exceeded |

---

## Terrain System

### Internal Grid Codes (what the API returns in `grid`)

| Code | Terrain | Prediction Class | Notes |
|---|---|---|---|
| 10 | Ocean | 0 (Empty) | **STATIC** — impassable water, borders map |
| 11 | Plains | 0 (Empty) | Flat land, buildable |
| 0 | Empty | 0 (Empty) | Generic empty cell |
| 1 | Settlement | 1 | Active Norse settlement |
| 2 | Port | 2 | Coastal settlement with harbour |
| 3 | Ruin | 3 | Collapsed settlement |
| 4 | Forest | 4 | Provides food to adjacent settlements |
| 5 | Mountain | 5 | **STATIC** — impassable terrain, never changes |

### Prediction Classes (what we submit)

| Index | Class | Description |
|---|---|---|
| 0 | Empty | Ocean (10), Plains (11), or Empty (0) all map here |
| 1 | Settlement | Active settlement |
| 2 | Port | Coastal settlement with harbour |
| 3 | Ruin | Collapsed settlement |
| 4 | Forest | Forest terrain |
| 5 | Mountain | Mountain terrain |

---

## 🔑 KEY STRATEGIC INSIGHTS

### 1. Static Cells = Free High-Confidence Predictions
- **Mountains (code 5)**: Never change. Initial state = final state. Predict class 5 with ~0.99.
- **Ocean (code 10)**: Never change. Predict class 0 with ~0.99.
- These can be read from `initial_states` at **zero query cost**.

### 2. Initial States Are Free
`GET /rounds/{round_id}` returns the full initial grid + settlement positions for all 5 seeds. No queries used.
- Identify all Mountain + Ocean cells immediately → near-certain predictions for ~30-50% of map
- Identify all initial settlement positions → focus query budget here

### 3. Ground Truth Is Itself a Probability Distribution
The ground truth (used for scoring) is computed from **Monte Carlo simulations** — it's a H×W×6 tensor of probabilities, not a deterministic outcome.
- For static cells (Mountain, Ocean): ground truth is near [0,0,0,0,0,1] or [1,0,0,0,0,0]
- For dynamic cells: ground truth captures the full stochastic distribution
- **Our queries are Monte Carlo samples** — each query gives one stochastic outcome

### 4. Budget Is Checkable Via API
`GET /budget` returns `queries_used` and `queries_max`. No need to track locally from scratch — always sync with API before querying.

### 5. Simulation Has Settlement Full Stats (via queries)
The `simulate` response includes settlement `population`, `food`, `wealth`, `defense`, `has_port`, `owner_id` — rich signal beyond just terrain type.

---

## ⚠️ CRITICAL SCORING RULES

### Exact Score Formula
```
KL(p || q)    = Σᵢ pᵢ × log(pᵢ / qᵢ)       # per cell; p=ground truth, q=our prediction
entropy(cell) = -Σᵢ pᵢ × log(pᵢ)            # cell weight

weighted_kl   = Σ_cells [entropy(cell) × KL(cell)]
                ──────────────────────────────────
                      Σ_cells entropy(cell)

score = max(0, min(100, 100 × exp(-3 × weighted_kl)))
```
- **100** = perfect match. **0** = terrible.
- Uniform `[1/6 × 6]` → score ≈ **1–5** out of 100.
- Static cells (Mountain, Ocean) have ~zero entropy → **excluded from scoring weight**.
- High-uncertainty dynamic cells are weighted **most heavily**.

### Never Assign 0.0 Probability
If ground truth `pᵢ > 0` but your prediction `qᵢ = 0` → `log(pᵢ / 0) = ∞` → **that cell = infinity KL**.

**Always apply floor BEFORE submitting:**
```python
prediction = np.maximum(prediction, 0.01)
prediction = prediction / prediction.sum(axis=-1, keepdims=True)
```

### Always Submit All 5 Seeds
Missing seed = **score 0** for that seed. Even a uniform prediction beats 0.
Round score = average of all 5 seed scores.

### Leaderboard Weighting — Later Rounds Count More
> "Leaderboard = best round score across all rounds — later rounds may have higher weight."

**Strategic implication:** Round 1 is for learning. Use it to:
- Confirm API response format
- Collect ground truth via `/analysis` post-round
- Build the transition matrix
- Tune α and Layer C boost weights

Don't sacrifice learning for a marginally better Round 1 score.

### Prediction Format Constraints
- `prediction[y][x]` = 6 floats summing to `1.0 ± 0.01`
- All values must be non-negative
- Shape: `H × W × 6` = `40 × 40 × 6`

---

## Simulation Mechanics

### Phases (each year, in order)
1. **Growth** — settlements produce food from adjacent terrain; grow population; develop ports; found new settlements on nearby land
2. **Conflict** — settlements raid each other; longships extend raiding range; desperate settlements raid more aggressively
3. **Trade** — ports within range trade if not at war; generates wealth + food; diffuses technology
4. **Winter** — all settlements lose food; collapse from starvation/raids/harsh winter → become Ruins
5. **Environment** — ruins reclaimed by thriving neighbours (possibly as Port); unclaimed ruins → forest or plains

### Settlement Properties (visible through queries)
`population`, `food`, `wealth`, `defense`, `tech_level`, `has_port`, `longship`, `owner_id (faction)`

Initial states only expose: `position` + `has_port` + `alive`. Internal stats require simulation queries.

### What Can Change

| Initial Terrain | Can Become |
|---|---|
| Settlement (1) | Settlement, Port, Ruin, possibly Forest/Empty if abandoned |
| Port (2) | Port, Settlement, Ruin |
| Forest (4) | Forest (mostly stable), can reclaim Ruins |
| Plains/Empty (0/11) | Empty, Settlement (if expansion), Forest |
| Mountain (5) | **Always Mountain** |
| Ocean (10) | **Always Ocean** |
| Ruin (3) | Ruin, Settlement, Port (if rebuilt), Forest/Empty (if abandoned) |

---

## API Reference

### Auth
```
Authorization: Bearer <JWT from app.ainm.no browser cookies>
```

### Endpoints

| Method | Path | Cost | Description |
|---|---|---|---|
| GET | `/rounds` | Free | List all rounds |
| GET | `/rounds/{round_id}` | **Free** | Full round details + **initial states for all seeds** |
| GET | `/budget` | Free | Check queries_used / queries_max |
| POST | `/simulate` | **1 query** | Observe viewport for one seed |
| POST | `/submit` | Free | Submit H×W×6 prediction tensor |
| GET | `/my-rounds` | Free | Your scores, rank, budget per round |
| GET | `/my-predictions/{round_id}` | Free | Your predictions (argmax + confidence grids) |
| GET | `/analysis/{round_id}/{seed_index}` | Free | Post-round: your prediction vs ground truth |
| GET | `/leaderboard` | Free | Public leaderboard |

### GET /rounds/{round_id} Response
```json
{
  "map_width": 40, "map_height": 40, "seeds_count": 5,
  "initial_states": [
    {
      "grid": [[10, 10, 0, ...], ...],
      "settlements": [{"x": 5, "y": 12, "has_port": true, "alive": true}]
    }
  ]
}
```

### GET /budget Response
```json
{"round_id": "uuid", "queries_used": 23, "queries_max": 50, "active": true}
```

### POST /simulate Request
```json
{
  "round_id": "uuid",
  "seed_index": 0,
  "viewport_x": 0, "viewport_y": 0,
  "viewport_w": 15, "viewport_h": 15
}
```

### POST /simulate Response
```json
{
  "grid": [[4, 11, 1, ...], ...],
  "settlements": [{"x": 12, "y": 7, "population": 2.8, "food": 0.4,
                   "wealth": 0.7, "defense": 0.6, "has_port": true,
                   "alive": true, "owner_id": 3}],
  "viewport": {"x": 0, "y": 0, "w": 15, "h": 15},
  "width": 40, "height": 40,
  "queries_used": 24, "queries_max": 50
}
```
⚠️ `width`/`height` = **full map dimensions** (always 40×40), NOT the viewport size.
`grid` contains only the viewport cells — shape is `viewport_h × viewport_w`.
Use `viewport.x` and `viewport.y` to place cells back on the full 40×40 grid.

### POST /submit Request
```json
{
  "round_id": "uuid",
  "seed_index": 0,
  "prediction": [[[0.85, 0.05, 0.02, 0.03, 0.03, 0.02], ...], ...]
}
```
Resubmitting overwrites the previous prediction. Only last submission counts.

### GET /analysis/{round_id}/{seed_index} — POST-ROUND ONLY
```json
{
  "prediction": [[[...], ...], ...],
  "ground_truth": [[[...], ...], ...],
  "score": 78.2,
  "initial_grid": [[10, 10, ...], ...]
}
```
Ground truth is the Monte Carlo H×W×6 probability distribution.

### Error Codes
| Code | Meaning |
|---|---|
| 400 | Round not active / invalid seed_index / round not completed (for analysis) |
| 403 | Not on a team |
| 404 | Round not found |
| 429 | Budget exhausted OR rate limit hit |

---

## Open Questions

- [ ] What hidden parameters govern simulation dynamics? (expansion rate, conflict rate, winter severity?)
- [ ] How frequently do terrain transitions occur? (quantify Settlement → Ruin rate)
- [ ] Which areas of the map are most dynamic (high entropy in ground truth)?
- [ ] Can we exploit `owner_id` faction data to predict future conflict zones?
- [ ] Does initial `has_port=True` almost always remain Port at year 50?
