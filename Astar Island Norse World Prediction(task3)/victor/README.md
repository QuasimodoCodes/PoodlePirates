
```
╔═══════════════════════════════════════════════════════════════════════════════════╗
║                                                                                   ║
║    ██╗   ██╗██╗██╗  ██╗██╗███╗   ██╗ ██████╗                                      ║
║    ██║   ██║██║██║ ██╔╝██║████╗  ██║██╔════╝                                      ║
║    ██║   ██║██║█████╔╝ ██║██╔██╗ ██║██║  ███╗                                     ║
║    ╚██╗ ██╔╝██║██╔═██╗ ██║██║╚██╗██║██║   ██║                                     ║
║     ╚████╔╝ ██║██║  ██╗██║██║ ╚████║╚██████╔╝                                     ║
║      ╚═══╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝ ╚═════╝                                      ║
║                                                                                   ║
║           N O R S E   W O R L D   P R E D I C T I O N                             ║
║                    Team: Poodle Pirates  |  Author: Victor                        ║
╚═══════════════════════════════════════════════════════════════════════════════════╝
```

---

## THE PROBLEM

```
┌─────────────────────────────────────────────────────────────────┐
│  A black-box Norse civilisation simulator runs for 50 YEARS.    │
│  Settlements grow. Factions clash. Forests reclaim ruins.       │
│  Winters wipe out entire civilisations.                         │
│                                                                 │
│  You cannot see inside.                                         │
│  You cannot replay it.                                          │
│  You get 50 questions. That's it.                               │
│                                                                 │
│  GOAL: Predict the probability distribution of terrain          │
│        across the ENTIRE map — for each of 5 parallel worlds.   │
└─────────────────────────────────────────────────────────────────┘
```

---

## THE WORLD

```
  THE FULL MAP (40 × 40 = 1,600 cells)
  ┌────────────────────────────────────────┐
  │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│  ▓ = Ocean   (static, class 0)
  │▓▓░░░░░░░░░░░▲▲▲░░░░░░░░░░░░░░░░░░░░▓▓│  ░ = Plains  (dynamic, class 0)
  │▓▓░░░S░░░░░░▲▲▲▲▲░░░░░░░░F░░░F░░░░░░▓▓│  ▲ = Mountain(static, class 5)
  │▓▓░░░░░░░░░░░▲▲▲░░░░░░░░░F░F░F░░░S░░▓▓│  F = Forest  (mostly static)
  │▓▓░░░░S░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▓▓│  S = Settlement (dynamic!)
  │▓▓░░░░░░░░░░░░░░░░░░░P░░░░░░░░░░░░░░▓▓│  P = Port    (dynamic!)
  │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│
  └────────────────────────────────────────┘
         ↑ THIS is what we must predict
```

### 6 Terrain Classes (what we predict)

```
  INDEX │ CLASS      │ INTERNAL CODES   │ BEHAVIOR
  ──────┼────────────┼──────────────────┼────────────────────────────────
    0   │ Empty      │ 0, 10, 11        │ Open land / Ocean / Plains
    1   │ Settlement │ 1                │ Active Norse settlement
    2   │ Port       │ 2                │ Coastal settlement with harbour
    3   │ Ruin       │ 3                │ Collapsed settlement
    4   │ Forest     │ 4                │ Provides food to neighbours
    5   │ Mountain   │ 5                │ STATIC — never changes
```

---

## THE HARD CONSTRAINTS

```
  ╔══════════════════════════════════════════════════════╗
  ║  ⚠️  BUDGET: 50 QUERIES TOTAL — SHARED ACROSS ALL   ║
  ║              5 SEEDS — CANNOT BE UNDONE              ║
  ╚══════════════════════════════════════════════════════╝

  MAP SIZE        │  40 × 40 = 1,600 cells
  VIEWPORT MAX    │  15 × 15 = 225 cells per query
  SEEDS           │  5 parallel worlds per round
  QUERIES/SEED    │  ~10  (50 ÷ 5)
  SIMULATION TIME │  50 years per run
  RATE LIMIT      │  5 requests/second (simulate)
  SCORE RANGE     │  0 → 100  (higher is better)
  UNIFORM SCORE   │  ~1–5 / 100  (naive baseline)
```

---

## THE SCORING

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                 │
  │  Ground Truth = Monte Carlo average of HUNDREDS of sim runs     │
  │  → It's a probability distribution, not a single answer         │
  │                                                                 │
  │  KL(p ║ q) = Σᵢ pᵢ × log(pᵢ / qᵢ)   ← per cell                  │
  │                                                                 │
  │  weighted_kl = Σ [entropy(cell) × KL(cell)]                     │
  │                ─────────────────────────────                    │
  │                      Σ entropy(cell)                            │
  │                                                                 │
  │  score = max(0, min(100,  100 × exp(-3 × weighted_kl)))         │
  │                                                                 │
  │  ⚡ Static cells (Mountain, Ocean) → entropy ≈ 0 → EXCLUDED    │
  │  ⚡ High-uncertainty dynamic cells → weighted MOST             │
  │                                                                │
  │  ☠️  NEVER assign 0.0 — if ground truth pᵢ > 0 and             │
  │      your qᵢ = 0 → log(pᵢ/0) = ∞ → cell score DESTROYED         │
  │                                                                 │
  │  FIX:  prediction = np.maximum(prediction, 0.01)                │
  │        prediction /= prediction.sum(axis=-1, keepdims=True)     │
  └─────────────────────────────────────────────────────────────────┘
```

---

## OUR SOLUTION

### Step 1 — Free Intelligence (costs 0 queries)

```
  GET /rounds/{id}  →  initial_states for ALL 5 seeds
                        ↓
  ┌───────────────────────────────────────────────────┐
  │  For each seed, classify every cell:              │
  │                                                   │
  │  Code 5  (Mountain) → STATIC → predict class 5   │
  │  Code 10 (Ocean)    → STATIC → predict class 0   │
  │  Code 1,2 (Settlement/Port) → DYNAMIC → observe  │
  │  Code 0,11 (Empty/Plains)   → DYNAMIC → observe  │
  │  Code 4  (Forest)           → DYNAMIC → observe  │
  └───────────────────────────────────────────────────┘
  Result: up to ~40% of cells predicted FREE with near-certainty
```

### Step 2 — Strategic Observation (costs 45 queries)

```
  TILE GRID STRATEGY: anchors at [0, 15, 25]
  ┌──────────────────────────────────────────┐
  │ (0,0)──────(15,0)──────(25,0)            │
  │   │  tile1  │  tile2  │  tile3  │        │
  │ (0,15)────(15,15)────(25,15)             │
  │   │  tile4  │  tile5  │  tile6  │        │
  │ (0,25)────(15,25)────(25,25)             │
  │   │  tile7  │  tile8  │  tile9  │        │
  │                                   40×40  │
  └──────────────────────────────────────────┘
  9 tiles × 5 seeds = 45 queries → full map observed every seed
  Note: anchors at 25 (not 30) so tiles stay within map bounds

  5 spare queries → re-query highest-uncertainty dynamic zones
```

### Step 3 — Build Probability Distributions

```
  For each cell (y, x) across all 5 seeds:

  STATIC  cell  →  [1e-5, 1e-5, 1e-5, 1e-5, 1e-5, ~1.0]  (Mountain example)
                    apply 1e-5 floor → renormalize

  DYNAMIC cell  →  count terrain class across all queries for this cell
                   empirical frequency = our Monte Carlo estimate
                   apply 0.01 floor → renormalize

  OUTPUT: 5 × 40 × 40 × 6  probability tensors
          └─seeds  └─H  └─W  └─classes
```

### Step 4 — Submit

```
  POST /submit  ×5  →  one per seed
  Check leaderboard on app.ainm.no

  score > previous best?
    YES → open PR: victor → main (ask user to confirm)
    NO  → go back, improve model, resubmit
```

---

## THE PIPELINE

```
  main.py
    │
    ├─[1]─ Auth + GET /rounds  →  find active round_id
    │
    ├─[2]─ GET /rounds/{id}    →  initial_states (FREE)
    │         │
    │         └─ initial_analyzer.py  →  classify all 1600 cells per seed
    │
    ├─[3]─ GET /budget         →  confirm queries_used before touching budget
    │
    ├─[4]─ query_planner.py    →  generate 45 (seed, x, y, 15, 15) viewport plans
    │
    ├─[5]─ runner.py           →  execute queries one-by-one
    │         │                   save JSON to data/observations/ after EACH call
    │         └─ rate limit: sleep(0.2) between calls  →  max 5 req/sec
    │
    ├─[6]─ terrain_estimator.py →  per-cell empirical frequency distributions
    │
    ├─[7]─ tensor_builder.py    →  assemble 40×40×6 tensors, assert sums = 1.0
    │
    └─[8]─ POST /submit ×5     →  done, check leaderboard
```

---

## FILE STRUCTURE

```
victor/
├── README.md               ← you are here
├── PROBLEM.md              ← full problem spec, API docs, scoring formula
├── CLAUDE.md               ← instructions for the AI assistant
├── plan.json               ← 12-step execution plan with statuses
├── config.py               ← ALL parameters live here (single source of truth)
├── requirements.txt
├── .env                    ← JWT token (gitignored, never committed)
│
├── src/
│   ├── api/
│   │   ├── client.py       ← ALL HTTP calls go through here
│   │   └── models.py       ← typed dataclasses for API requests/responses
│   ├── observation/
│   │   ├── query_planner.py ← generates (seed, x, y, w, h) viewport list
│   │   └── runner.py        ← executes plan, saves raw JSON per query
│   ├── model/
│   │   ├── initial_analyzer.py  ← classifies cells from initial_states (free)
│   │   └── terrain_estimator.py ← builds per-cell probability distributions
│   └── prediction/
│       └── tensor_builder.py    ← assembles + validates 40×40×6 tensors
│
├── data/
│   ├── observations/       ← raw API responses (seed{n}_x{x}_y{y}.json)
│   └── predictions/        ← final tensors (seed{n}_tensor.npy + .json)
│
├── scripts/
│   ├── api_discovery.py    ← one-off probe: reads free endpoints, 0 queries
│   └── analyze_observations.py  ← inspect coverage, terrain frequency, variance
│
├── problems_log.md         ← short issue summaries (1-2 sentences each)
└── features_backlog.md     ← future improvements (transition matrix, Bayesian model)
```

---

## GIT RULES

```
  Branch:  victor  (all work here)
  Remote:  NO push until confirmed better score
  Main:    ONLY merge when score > previous best — always ask first

  Commit format:
    feat: Complete step X - description
    fix:  description
    chore: description
```

---

## KEY REMINDERS

```
  ┌──────────────────────────────────────────────────────────┐
  │  1. NEVER assign 0.0 probability — use floors always     │
  │  2. Tile anchors are [0, 15, 25] — NOT [0, 15, 30]      │
  │  3. simulate() returns YEAR-50 state, not year-0         │
  │  4. initial_states are FREE — read them first always     │
  │  5. Save observation JSON after EVERY single query       │
  │  6. Always submit all 5 seeds — missing = score 0        │
  │  7. Check GET /budget before starting observations       │
  └──────────────────────────────────────────────────────────┘
```
