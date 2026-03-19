# Features Backlog

## High Priority

### Entropy-Guided Query Placement
After the initial full-map scan (9 tiles), use the 5 spare queries to target highest-uncertainty dynamic cells — cells that varied most across first observations. Maximizes information per remaining query.

### Cross-Seed Pattern Learning
Same hidden parameters govern all 5 seeds. If a cell type shows consistent behavior across seeds (e.g. settlements near mountains always collapse), exploit that to improve predictions on unobserved cells.

### Initial Settlement Survival Model
Use initial state data (position, `has_port`, proximity to ocean/forest/mountain) as features to predict whether a settlement survives, grows to Port, or collapses to Ruin by year 50.

### Post-Round Ground Truth Analysis
After each round, call `GET /analysis/{round_id}/{seed_index}` to get the real ground truth H×W×6 tensor. Compare cell-by-cell. Identify systematic errors (e.g. always under-predicting Ruin). Feed into next round's model.

---

## Medium Priority

### Terrain Transition Matrix
Track how often each initial terrain code (0/1/2/3/4/5/10/11) becomes each final class (0-5). Build from observations. Use as prior for unobserved dynamic cells.

### Settlement Cluster & Faction Analysis
Group initial settlements by `owner_id` faction from simulate responses. Friendly clusters may expand reliably; isolated ones are more raid-prone.

### Visualization Tool
ASCII or matplotlib grid showing: initial state, observed final states per seed, coverage map, confidence heatmap. Helps debug and spot patterns.

---

## Low Priority / Future Rounds

### MCP Server Integration
`claude mcp add --transport http nmiai https://mcp-docs.ainm.no/mcp`
Adds the platform docs to Claude Code for AI-assisted development.

### Bayesian Per-Cell Distribution (Dirichlet-Multinomial)
Treat each simulate query as a stochastic sample. Use conjugate prior model to update per-cell distributions as more samples arrive — more principled than raw frequency counting.

### Adaptive Query Budget Allocation
Allocate more budget to seeds with more dynamic/uncertain zones rather than fixed 10 per seed.
