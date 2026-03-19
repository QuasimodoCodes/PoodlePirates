# Claude Instructions — Astar Island Norse World Prediction

Read [PROBLEM.md](PROBLEM.md) before doing anything. It contains the full problem description, constraints, scoring rules, and open questions for this challenge.

---

## Workflow

You are a thoughtful and disciplined coding assistant. Follow these strict steps for structured code generation:

### 1. Project Folder Setup
- Before starting, create a dedicated folder named after the project (e.g., `my_project_plan`) to store all planning, logs, and progress files.
- This folder will be the central workspace where we track tasks, problems, decisions, and new ideas over time.

### 2. Plan the Task
- Before writing any code, break the task into a logical, step-by-step plan.
- Each step must be small, independent, and lead toward the goal.
- Output the plan as a JSON object with the following fields:
  - `"step_number"`
  - `"description"`
  - `"status"`: `"pending"` | `"in_progress"` | `"complete"`

### 3. Save the Plan
- Store the plan in a JSON file named `plan.json` inside the project folder.
- All other supporting files (logs, notes, feature ideas) can be saved in any format (e.g., Markdown, text files).

### 4. Iterative Execution
- Pick the first `"pending"` step.
- Update its status in `plan.json` to `"in_progress"`.
- Write only the code for that step.
- Suggest the appropriate command(s) to run the program.
- Instruct the user to run the program and describe what they should see or check for.

### 5. Verify Output
- Ask the user to confirm whether the step works as expected.
- If yes: mark the step `"complete"` in `plan.json` and continue.
- If not:
  - Log the problem and attempted solution in `problems_log.md` (1–2 sentences max).
  - Offer debug suggestions and pause execution until it's fixed.

### 6. Plan Revisions and Feature Expansion
- At any point, if new feature ideas arise, unexpected problems require new tasks, or improvements are identified:
  - Pause execution.
  - Update `plan.json` with new or revised steps.
  - Optionally update a `features_backlog.md` file.
  - Clearly explain why the plan was updated.

### 7. Continuous Logging
Throughout the project, maintain in the project folder:
- `problems_log.md` — short summaries of issues and fixes
- `features_backlog.md` — future improvement ideas
- Any other relevant notes

### 8. Git Commit
After each verified, working step, commit using:
```
feat: Complete step X - [step description]
```

### 9. Repeat
Move to the next step and repeat the cycle until all steps are marked complete.

---

## Rules
- Only code one step at a time.
- Always ask for output verification before moving on.
- Never skip steps.
- Always commit verified changes.
- Keep the plan in JSON format and maintain logs as short summaries.

## Git Branching
- All work is done on the `victor` branch.
- **Never merge to `main` unless we have confirmed better results** (higher score / lower KL divergence than the previous submission).
- Before merging, explicitly verify the score improvement and ask the user to confirm.
