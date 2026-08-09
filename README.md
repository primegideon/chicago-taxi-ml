# Project 4 — Team 2 — Ride-Share Demand Analysis

**Organisation:** data-analytics-internship-2026
**Domain:** Transport
**Difficulty:** Intermediate
**Competing against:** Project 4 — Team 1 (same dataset, head-to-head)

---

## Crown Metric Question

> What is the most profitable hour-and-day combination for a Chicago taxi driver?

The group with the stronger, more data-supported answer wins the project crown (+10 bonus points).
Rival group visibility opens after the **Week 5 checkpoint** — until then you work blind.

---

## BigQuery Dataset

```
bigquery-public-data.chicago_taxi_trips.taxi_trips
```


---

## Group Members

| Name | Role | Track |
|------|------|-------|
| Gloria  | Project Manager | — |
| Gideon  | Technical Deputy | — |
| Rolland & Yaaziah  | Data/Python Members | Data track |
| Rolland & Belinda  | SQL Members | Data track |
| Moses & Dela  | BI Members | BI track |

---

## Repo Structure

```
project-4-team-2/
├── notebooks/          # Google Colab EDA notebooks — commit .ipynb files here
├── sql/                # All .sql query files — at least 8 required
├── src/                # Shared Python utility functions (optional)
├── reports/            # Written report (PDF or Word) committed here
├── slides/             # Slide deck committed here
├── dashboard_link.md   # Paste your Looker Studio or Power BI share link here
├── known_issues.md     # Data quality issues and EDA surprises
└── README.md           # This file
```

---

## Deliverables Tracker

| Deliverable | Owner | Status | Due |
|-------------|-------|--------|-----|
| EDA Notebook | Data/Python members | Started | Week 6 |
| SQL Query Library (8+ queries) | SQL members | Started | Week 6 |
| Dashboard (live BigQuery) | BI members | started | Week 6 |
| Written Report | All | started | Week 7 |
| Slide Deck | All | started | Week 7 |

---

## Tool Decision

**Dashboard tool:** [ ✓ ] Looker Studio [ ]  Power BI

**Reason:** Its seamless and native connectivity to BigQuerry. No installations and requires less technical prior knowledge

---

## BigQuery Config

| Setting | Value |
|---------|-------|
| GCP Project ID | ride-share-demand-analysis |
| BigQuery Dataset | bigquery-public-data.chicago_taxi_trips.taxi_trips |
| Billing alert set | [ ✓ ] Yes |
| Tool connected to BQ | [ ✓ ] Yes |

---

## Week 1 Checklist

- [ ✓ ] BigQuery console access confirmed — dataset found and previewed
- [ ✓ ] Billing alert set at $10
- [ ✓ ] Colab notebook created and BigQuery query tested
- [ ✓ ] Dashboard tool decided and BigQuery connection confirmed
- [ ✓ ] GitHub repo access confirmed for all members
- [ ✓ ] Project guide read — EDA questions noted in notebook
- [ ✓ ] Group kickoff meeting held — roles confirmed, weekly standup scheduled

---

## API Contract (SQL Query Interface)

> Document your key SQL query patterns here so all team members write consistent queries.





**Primary table filter:**
```
-- Always start queries with this filter to control scan cost
SELECT
      specific_columns
FROM
    `bigquery-public-data.chicago_taxi_trips.taxi_trips`
WHERE
    trip_start_timestamp >= '2023-01-01' AND trip_start_timestamp < '2024-01-01'
LIMIT 1000  -- remove for final analysis queries
```

---

## MLflow Equivalent — Query Run Log

Track your significant analysis queries here so the team doesn't duplicate work.

| Query file | What it answers | Run date | Result summary |
|------------|----------------|----------|----------------|
| sql/01_overview.sql | | | |


