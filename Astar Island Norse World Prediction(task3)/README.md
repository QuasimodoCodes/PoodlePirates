# Astar Island - Viking Civilisation Prediction

## Project Setup Complete ✓

### What's Ready:
- **requirements.txt** - Python dependencies (requests, numpy, pandas, python-dotenv)
- **.env.example** - Template for configuration (copy to .env and fill in your values)
- **src/** - Source code directory for API client, data parsers, strategies
- **data/** - Directory for storing observations and predictions
- **logs/** - Directory for tracking queries and debugging information

### Next Steps:
1. Create virtual environment: `python3 -m venv venv`
2. Activate it: `source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Create `.env` file from `.env.example` with your Google auth token
5. Run next step: API client creation

### Project Budget:
- **Total Queries**: 50 per round (shared across all 5 seeds)
- **Map Size**: 40×40 cells
- **Viewport**: 15×15 cells max per query
- **Seeds**: 5 different stochastic runs

### Plan Status:
See `plan.json` for detailed step tracking and progress updates.
