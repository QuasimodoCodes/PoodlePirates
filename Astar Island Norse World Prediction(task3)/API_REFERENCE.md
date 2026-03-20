# Astar Island API Reference

**Base URL**: `https://api.ainm.no/astar-island`

**Authentication**: Bearer token in `Authorization: Bearer <token>` header

**Rate Limits**:
- POST /simulate: 5 req/sec per team
- POST /submit: 2 req/sec per team

---

## Key Endpoints

### POST /simulate - Query Simulator
**Budget**: 1 query per call (50 total per round)

**Request**:
```json
{
  "round_id": "uuid",
  "seed_index": 0,          // 0-4
  "viewport_x": 0,          // 0-39
  "viewport_y": 0,          // 0-39
  "viewport_w": 15,         // 5-15
  "viewport_h": 15          // 5-15
}
```

**Response**:
```json
{
  "grid": [[4, 11, 1, ...], ...],    // viewport_h × viewport_w
  "settlements": [
    {
      "x": 12,
      "y": 7,
      "population": 2.8,
      "food": 0.4,
      "wealth": 0.7,
      "defense": 0.6,
      "has_port": true,
      "alive": true,
      "owner_id": 3
    }
  ],
  "viewport": {"x": 10, "y": 5, "w": 15, "h": 15},
  "width": 40,
  "height": 40,
  "queries_used": 24,
  "queries_max": 50
}
```

---

### POST /submit - Submit Predictions

**Request**:
```json
{
  "round_id": "uuid",
  "seed_index": 0,      // 0-4
  "prediction": [       // H × W × 6 tensor
    [
      [0.85, 0.05, 0.02, 0.03, 0.03, 0.02],
      ...
    ],
    ...
  ]
}
```

**Prediction Tensor Format**:
- `prediction[y][x][class]` = probability for class at cell (x, y)
- Height × Width × 6 classes
- Each cell's 6 probabilities must sum to 1.0 (±0.01 tolerance)
- All values must be non-negative

**Class Indices**:
- 0 = Empty (Ocean, Plains)
- 1 = Settlement
- 2 = Port
- 3 = Ruin
- 4 = Forest
- 5 = Mountain

---

### GET /budget - Check Query Budget

**Response**:
```json
{
  "round_id": "uuid",
  "queries_used": 23,
  "queries_max": 50,
  "active": true
}
```

---

### GET /rounds - List All Rounds

**Response**: Array of round objects with status, dates, map size, etc.

---

### GET /rounds/{round_id} - Round Details + Initial States

**Response**: Returns round info + initial terrain grids for all 5 seeds

---

### GET /my-rounds - Your Team's Rounds (Auth Required)

**Response**: Array of rounds with your team's scores, submissions, budget, rank

---

### GET /my-predictions/{round_id} - Your Predictions (Auth Required)

**Response**: Array of predictions with argmax grid, confidence grid, score

---

### GET /analysis/{round_id}/{seed_index} - Post-Round Analysis (Auth Required)

**Response**: Your prediction vs ground truth for detailed comparison

---

## Grid Cell Values

| Value | Meaning |
|-------|---------|
| 0 | Empty (Ocean, Plains) |
| 1 | Settlement |
| 2 | Port |
| 3 | Ruin |
| 4 | Forest |
| 5 | Mountain |
| 10 | Ocean |
| 11 | Plains |

---

## Error Codes

**Common Errors**:
- `400` - Invalid request (bad seed_index, inactive round, etc.)
- `403` - Not on a team / unauthorized
- `404` - Round not found
- `429` - Budget exhausted or rate limit exceeded

---

## Important Notes

1. **Map Size**: Always 40×40
2. **Viewport**: Always observe through 5-15 × 5-15 windows (never full map directly)
3. **Stochastic**: Same map + params = different outcome each query
4. **Budget**: 50 queries total, shared across all 5 seeds
5. **Submission**: You must submit all 5 seeds for scoring
6. **Resubmit**: Later submissions overwrite earlier ones for the same seed
7. **Scoring**: Entropy-weighted KL divergence between your prediction and ground truth
