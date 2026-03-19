# Astar Island — Viking Civilisation Prediction

## Overview

Observe a black-box Norse civilisation simulator through a limited viewport and predict the final world state.

The simulator runs a procedurally generated Norse world for **50 years** — settlements grow, factions clash, trade routes form, alliances shift, forests reclaim ruins, and harsh winters reshape entire civilisations.

**Goal:** Observe through a viewport, learn the world's hidden rules, and predict the probability distribution of terrain types across the entire map.

- **Task type:** Observation + probabilistic prediction
- **Platform:** app.ainm.no
- **API base:** `https://api.ainm.no/astar-island/`

---

## How It Works

1. A round starts — admin creates a round with a fixed map, hidden parameters, and **5 random seeds**
2. **Observe** — call `POST /astar-island/simulate` with viewport coordinates to observe one stochastic run through a window (max **15×15 cells**). You have **50 queries total** per round, shared across all 5 seeds.
3. **Learn** — analyze viewport observations to understand the forces governing the world
4. **Predict** — build probability distributions for the full map
5. **Submit** — for each seed, submit a **H×W×6 probability tensor** (terrain type probabilities per cell)
6. **Score** — prediction is compared to ground truth using **entropy-weighted KL divergence**

---

## Key Constraints

| Parameter | Value |
|---|---|
| **TOTAL_QUERIES** | **50** — shared across ALL 5 seeds |
| **QUERIES_PER_SEED** | ~10 (50 / 5) |
| **MAP_SIZE** | 40×40 = 1,600 cells |
| **VIEWPORT_MAX** | 15×15 = 225 cells per query |
| **TILES_TO_COVER_MAP** | 9 (3×3 grid of 15×15 tiles) |
| **SPARE_QUERIES** | 5 (after full coverage of all seeds: 50 - 5×9 = 5) |
| **NUM_SEEDS** | 5 |
| **NUM_TERRAIN_CLASSES** | 6 |
| **SIMULATION_YEARS** | 50 |

---

## Terrain Classes (6 total) ✅ CONFIRMED

| Index | Terrain Type | Notes |
|---|---|---|
| 0 | Empty | Open land, no activity |
| 1 | Settlement | Viking settlement present |
| 2 | Port | Coastal settlement with port |
| 3 | Ruin | Destroyed/abandoned settlement |
| 4 | Forest | Forest has grown or reclaimed area |
| 5 | Mountain | Mountain terrain |

---

## ⚠️ CRITICAL SCORING WARNING

**Never assign `0.0` probability to any class.**

If the ground truth has any non-zero probability for a class you marked as `0.0`, KL divergence becomes **infinite** and your score for that cell is destroyed.

**Rule:** Always apply a minimum probability floor (e.g. `0.01`) to all classes, then renormalize so the row sums to `1.0`.

---

## Scoring

Metric: **Entropy-weighted KL divergence** — lower is better.

- Higher-uncertainty cells (high entropy ground truth) are weighted **more**
- Near-deterministic cells contribute less to the score
- A **uniform prediction** (`[1/6, 1/6, 1/6, 1/6, 1/6, 1/6]`) scores approximately **1–5**
- Confident wrong answers are penalized most severely → **never be overconfident**

---

## 🔑 KEY INSIGHT: Initial States Are Free

`GET /astar-island/rounds/{round_id}` returns **`initial_states`** for every seed at zero query cost.

Each initial state includes:
- `grid`: full `H×W` terrain grid at year 0
- `settlements`: list of `{x, y, has_port, alive}` for every settlement

**This means we know the starting map for free.** Our 50 queries should focus on understanding HOW the simulation transforms the initial state — not discovering what's there at year 0.

---

## API Reference ✅ CONFIRMED

### Authentication
Log in at app.ainm.no, grab `access_token` JWT from browser cookies.

```python
session = requests.Session()
session.headers["Authorization"] = "Bearer YOUR_JWT_TOKEN"
```

### Get Active Round
```
GET /astar-island/rounds
```
Returns list of rounds. Find the one where `status == "active"`.

### Get Round Details (FREE — costs no queries)
```
GET /astar-island/rounds/{round_id}
```
Returns:
- `map_width`, `map_height` (confirmed 40×40)
- `seeds_count` (confirmed 5)
- `initial_states[i]["grid"]` — full H×W terrain grid at year 0
- `initial_states[i]["settlements"]` — list of `{x, y, has_port, alive}`

### Simulate (Observe) — costs 1 query
```
POST /astar-island/simulate
{
  "round_id": "<round_id>",
  "seed_index": 0,          ← 0 to 4
  "viewport_x": 10,
  "viewport_y": 5,
  "viewport_w": 15,          ← max 15
  "viewport_h": 15           ← max 15
}
```
Returns:
- `grid`: H×W terrain after 50 years of simulation (the viewported window)
- `settlements`: settlements in viewport with full stats
- `viewport`: `{x, y, w, h}` (echo of your request)

### Submit Prediction
```
POST /astar-island/submit
{
  "round_id": "<round_id>",
  "seed_index": 0,
  "prediction": [[[p0,p1,p2,p3,p4,p5], ...], ...]   ← H×W×6, sums to 1.0 per cell
}
```

---

## Strategy Notes

- **Initial state is free** → fetch `initial_states` first, use as prior for every seed
- **50 queries / 5 seeds = ~10 per seed** — 9 for full map scan, 1 spare per seed
- 9 non-overlapping 15×15 tiles cover the full 40×40 map (3×3 grid)
- Cross-seed observations reveal stochasticity — same cell seen multiple times = probability estimate
- **Minimum probability floor**: apply `0.01` floor to all 6 classes, then renormalize
- Initial terrain + observed final terrain = training signal for extrapolating unobserved cells

---

## Open Questions

- [x] What are the 6 terrain types? **→ Empty, Settlement, Port, Ruin, Forest, Mountain**
- [x] What is the full API endpoint spec? **→ documented above**
- [x] What does the viewport response look like? **→ {grid, settlements, viewport}**
- [x] What's the map size? **→ always 40×40**
- [x] What is the auth method? **→ Bearer JWT from browser cookies**
- [ ] What hidden parameters govern simulation dynamics? (discover via observation)
- [ ] How frequently do terrain transitions occur? (e.g., Settlement → Ruin frequency)
- [ ] Does initial settlement `has_port` always become terrain class 2 (Port)?
- [ ] Which terrain types are deterministic vs stochastic at final state?
