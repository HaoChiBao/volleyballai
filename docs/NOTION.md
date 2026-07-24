# Notion project links

Source of truth for tickets and long-form wiki. Keep this file updated if URLs change.

## Hub

- [Volleyball AI — Project Wiki](https://app.notion.com/p/3a7850bd5b0881b3ba1afe969912b9f3)

## Docs in Notion

- [Ticket Board](https://app.notion.com/p/3a7850bd5b08813185cad8222b87df2b) — how to use tickets + Local v0 backlog
- [Tickets database](https://app.notion.com/p/b86adbc0f9294bf09e181b1e5a25e4c1) — views: Local v0, Board, By Milestone, P0 Now
- [Technical Architecture](https://app.notion.com/p/3a7850bd5b088179a775f2e6ca2bd67f)
- [Work Breakdown — Everything Needed](https://app.notion.com/p/3a7850bd5b0881b3a79beca1b8885064)

## Ticket fields

- **Milestone:** Local v0 · Local v1 · Cloud deploy · Later
- **Type:** Epic · Story · Task · Chore · Spike
- **Priority:** P0–P3
- **Workstream:** Infra, Ingest, Calibration, Players SAM, Ball, Actions, Score, 3D, Product, Eval
- **Acceptance:** short done criteria

## What to use when

| Need | Use |
|---|---|
| Day-to-day coding tasks | Notion **Tickets** → Local v0 |
| Locked architecture / policy | Repo `docs/` (this folder) |
| Deep diagrams | Notion Technical Architecture + repo `docs/ARCHITECTURE.md` |

Repo markdown should stay aligned with Notion decisions; if they diverge, update both.
