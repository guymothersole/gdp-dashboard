# AW GPS — Coach Dashboard

A small Streamlit app for exploring team and player GPS match-load data (distance, HSR, sprints, top speed, accelerations/decelerations) across matches, players and positions.

This repository contains a single-file Streamlit app (`streamlit_app.py`) that expects a CSV export with match-by-match player outputs. The app provides coach-friendly KPIs, match profiles, player trend views, position comparisons and data export.

Features

- Overview dashboard with season KPIs and automatic insights
- Match-level and player-level visualisations (bar charts, scatter, line charts, radar)
- Position comparison and per-minute metric boxplots
- Sprint and top-speed leaderboards
- Acceleration / deceleration summaries and flags for extreme values
- Filterable dataset with CSV download of the filtered subset

Getting started

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Provide your data

Place your CSV export at `data/AW_GPS_Data.csv` next to the app. The script includes a mapping/cleaning step and expects common column names; typical columns used are:

- `DATE`, `TIME`, `NAME`, `OPPOSITION`, `RESULT`, `FOR`, `AGAINST`
- Distance/metric columns like `TOTAL D (m)`, `HSR D (m) 19.8 - 25.2 kmh`, `SPRINT D (m) 25.2+ kmh`, `TOP SPEED (kmh)`, `ACCEL COUNT 2+ m/s`, `DECEL COUNT 2+ m/s`
- Per-minute columns: `Dist/min`, `HSR/min`, `Sprint D/min`, `Acc/min`, `Dec/min`

If key fields are missing (name, date, opposition, total distance) some features will be disabled; the app shows warnings about missing fields.

3. Run the app

```bash
streamlit run streamlit_app.py
```

Usage notes

- Use the sidebar to filter by date range, opposition, result, player and positions. The minute threshold slider helps flag short appearances.
- The app caches the cleaned CSV, so widget interactions are responsive without re-reading the file.
- Common opposition typos can be corrected via the `OPPOSITION_CORRECTIONS` mapping inside `streamlit_app.py`.

Files

- `streamlit_app.py` — main Streamlit application
- `requirements.txt` — Python dependencies
- `data/AW_GPS_Data.csv` — your data file (not included in the repo by default)

License

This project is provided as-is. See the repository for license details if present.
