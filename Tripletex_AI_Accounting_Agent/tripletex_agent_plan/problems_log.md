# Problems Log

_Short summaries of issues encountered and how they were resolved._

---
## 2026-03-19 - Gemini 2.5 Flash working

Issue: gemini-2.0-flash and gemini-1.5-flash unavailable for new users. gemini-2.0-flash-lite also removed.
Fix: Switched to gemini-2.5-flash after billing was enabled on Google Cloud project.
Status: Agent loop confirmed working - Gemini correctly called tripletex_post /employee with right payload.
Remaining: Tripletex session token expired during test - need fresh token from platform to verify full flow.

