# Learning Log — Prediction vs Reality

One entry per round. Max 5 bullet points. Focus on **surprises** — systematic errors, not random noise.
Each entry feeds directly into tuning α, Layer C boost weights, and eventually the transition matrix.

---

## Format

```
## Round {id} — Score: {X}/100 — Date: YYYY-MM-DD
- [WRONG] What we predicted vs what actually happened + which cells/classes
- [FIX]   What to change in terrain_estimator.py for next round
- [TUNE]  α was too high/low? Which Layer C boost weights need adjusting?
- [LEARN] Any new pattern discovered about the simulation dynamics
```

---

## Round History

### Pre-Round 4 — Transition Matrix from Rounds 1–3 — 2026-03-20
Source: 15 analysis files × 1600 cells = 24,000 cells of ground truth data

- [LEARN] Settlements COLLAPSE: 47.6% of initial Settlements become Empty after 50 years. Only 28.3% survive. Our assumption that "settlements mostly stay settlements" was completely wrong.
- [LEARN] Forests are very stable: 79.5% stay Forest. Only 11.8% get colonised into Settlements. Safe to predict high Forest confidence for isolated forest cells.
- [LEARN] Plains mostly stay Empty (83.2%) but 11.6% become Settlement — expansion from nearby settlements is the dominant dynamic force on the map.
- [LEARN] Ports collapse even more than Settlements: only 21% survive as Port. 47.2% become Empty, 21.1% become Forest.
- [FIX] Layer C now uses real transition matrix instead of hand-coded spatial rules. transition_matrix.json saved to data/.
- [RESULT] Offline test on all 15 ground truth files: avg=63.66, best=71.62, worst=51.19 — with ZERO queries. Uniform baseline ~1-5. Transition matrix is highly effective.
- [STRATEGY] Submit transition-matrix-only prediction first (free), then spend 50 queries for Bayesian updates and resubmit.
- [CV RESULT] Leave-one-round-out CV: 53.83 avg (range 38-67). Round f1dac9a9 scores 39 when excluded from training — its hidden parameters were significantly different. Round 4 expected 39-67 depending on parameters.
- [KEY INSIGHT] Bayesian update (50 queries) is most important when round has unusual parameters — observations adapt the prior to the actual dynamics of THIS round.

---

## Transition Matrix (built from round history)

Once we have 2+ rounds of data, fill this in from `GET /analysis` ground truths:

```
initial_code → [Empty, Settlement, Port, Ruin, Forest, Mountain]  (n=samples)

Code  1 (Settlement) → [0.476, 0.283, 0.004, 0.023, 0.214, 0.000]  n=637  ← mostly collapse!
Code  2 (Port)       → [0.472, 0.085, 0.210, 0.021, 0.211, 0.000]  n=26
Code  4 (Forest)     → [0.068, 0.118, 0.009, 0.010, 0.795, 0.000]  n=5029 ← very stable
Code  5 (Mountain)   → [0.000, 0.000, 0.000, 0.000, 0.000, 1.000]  n=461  ← static ✅
Code 10 (Ocean)      → [1.000, 0.000, 0.000, 0.000, 0.000, 0.000]  n=3180 ← static ✅
Code 11 (Plains)     → [0.832, 0.116, 0.010, 0.011, 0.031, 0.000]  n=14667

Note: Code 0 (Empty) and Code 3 (Ruin) not seen in initial states — treat as Plains prior.
Saved in full at: data/transition_matrix.json
```

When this table is filled, replace Layer C spatial rules with `transition_prior[initial_code]`.

---

## Round 4 Results — 2026-03-20

- [RESULT] Leaderboard score: **71.7 / 100** — rank #88. Above CV best of 67 and offline test best of 71.62.
- [LEARN] Round 4 dynamics matched transition matrix well. Settlement→Forest slightly higher (+4%). Port→Forest outlier but n=14 too small to trust.
- [LEARN] 50 Bayesian queries added ~8 points over zero-query baseline (~63.66). Full map scan confirmed effective.
- [FIX] Removed double-submission — query first, submit once.
- [NOTE] Round 4 ground truth (8e839974) still 401 from /analysis — retry after round 5.

---

## Round 5 Results — 2026-03-20

- [RESULT] Leaderboard score: **43.9 / 100** — rank #103 of 144. Major drop from round 4.
- [WRONG] Bayesian update (α=0.55) HURT by ~20 points. Offline test (no Bayesian) gives ~64 on round 5 data; actual with Bayesian = 43.9.
- [LEARN] Single simulate() observation is one stochastic run. High-variance rounds make this very misleading. Historical matrix average is more reliable than one noisy observation.
- [LEARN] Round 5 was likely high-variance (hidden parameters caused wider spread of outcomes). α=0.55 too aggressively trusted single observations.
- [FIX] Lowered α from 0.55 → 0.30. Transition matrix now built from 5 rounds (40,000 cells).

---

## α Calibration History

| Round | α used | Result | Next α |
|-------|--------|--------|--------|
| 4     | 0.55   | 71.7   | — |
| 5     | 0.55   | 43.9   | 0.30 ← too high, single obs misleading on high-variance rounds |
| 6     | 0.30   | pending | — |

---

## Layer C Boost Weight History

| Round | Rule | Boost used | Was it right? | Next value |
|-------|------|-----------|---------------|------------|
| 4     | Settlement adj. | +0.15 | yes — Settlement survival ~28% confirmed | keep |
| 4     | Coastal Port    | +0.20 | yes — Port cells rare, matrix holds | keep |
| 4     | Mountain block  | -0.10 | yes — Mountain 100% static | keep |
| 4     | Far from settlement | +0.15/+0.10 | yes — Plains→Empty 83-85% confirmed | keep |
| 4     | Isolated forest | +0.20 | yes — Forest 79-81% stable confirmed | keep |
