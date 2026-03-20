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

_(No rounds completed yet — entries will be added after each submission via scripts/post_round_analysis.py)_

---

## Transition Matrix (built from round history)

Once we have 2+ rounds of data, fill this in from `GET /analysis` ground truths:

```
initial_code → average final class distribution after 50 years

Code 0  (Empty)      → [?, ?, ?, ?, ?, ?]
Code 1  (Settlement) → [?, ?, ?, ?, ?, ?]  ← most important to get right
Code 2  (Port)       → [?, ?, ?, ?, ?, ?]
Code 3  (Ruin)       → [?, ?, ?, ?, ?, ?]
Code 4  (Forest)     → [?, ?, ?, ?, ?, ?]
Code 5  (Mountain)   → [0, 0, 0, 0, 0, 1]  ← static, confirmed
Code 10 (Ocean)      → [1, 0, 0, 0, 0, 0]  ← static, confirmed
Code 11 (Plains)     → [?, ?, ?, ?, ?, ?]
```

When this table is filled, replace Layer C spatial rules with `transition_prior[initial_code]`.

---

## α Calibration History

| Round | α used | Result | Next α |
|-------|--------|--------|--------|
| —     | 0.55   | pending | —     |

---

## Layer C Boost Weight History

| Round | Rule | Boost used | Was it right? | Next value |
|-------|------|-----------|---------------|------------|
| —     | Settlement adj. | +0.15 | pending | — |
| —     | Coastal Port    | +0.20 | pending | — |
| —     | Mountain block  | -0.10 | pending | — |
| —     | Far from settlement | +0.15/+0.10 | pending | — |
| —     | Isolated forest | +0.20 | pending | — |
