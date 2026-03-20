# Tripletex AI Accounting Agent

An AI-powered agent that ingests invoices and receipts (PDF, images, email attachments), extracts structured accounting data using Claude, and automatically posts vouchers to Tripletex.

## Architecture

```
inbox/ or IMAP email
        │
        ▼
  [Ingestion Layer]     src/ingestion/    — PDF & image → base64
        │
        ▼
  [AI Extraction]       src/extraction/   — Claude vision → Pydantic model
        │
        ▼
  [Validation]          src/validation/   — map to Tripletex schema
        │
        ▼
  [Tripletex Client]    src/tripletex/    — POST voucher via REST API
        │
        ▼
  [Logging]             logs/             — JSON run logs
```

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
cp .env.example .env            # fill in your tokens
python -m src.orchestrator.agent
```

## Running a single document

```bash
python -m src.orchestrator.agent --file inbox/invoice.pdf
```

## Environment Variables

See `.env.example` for all required variables.

## Project Layout

```
src/
  tripletex/      Tripletex REST API client
  ingestion/      PDF & image loader
  extraction/     Claude-based data extraction
  email_ingest/   IMAP email fetcher
  validation/     Schema validation & field mapping
  posting/        Tripletex voucher posting
  orchestrator/   Main agent loop
inbox/            Drop documents here for processing
logs/             JSON run logs
tests/            Pytest test suite
tripletex_agent_plan/  Planning files and logs
```
