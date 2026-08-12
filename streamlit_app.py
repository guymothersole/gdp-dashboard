"""
AW GPS — Coach Dashboard
Streamlit app for exploring GPS match-load data (distance, HSR, sprint, accel/decel)
across matches, players, and positions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="AW GPS — Coach Dashboard", page_icon=":soccer:", layout="wide")

CSS = """
<style>
.kpi-card {
    background: linear-gradient(180deg, #ffffff, #f7f9fc);
    border-radius: 10px;
    padding: 12px;
    box-shadow: 0 1px 3px rgba(16,24,40,0.06);
}
.kpi-title { color: #1f2937; font-size:14px; }
.kpi-value { color: #0f172a; font-weight:700; font-size:20px; }
.insight { background:#0f172a; color:#fff; padding:10px; border-radius:8px; }
.sidebar-note { font-size:12px; color:#6b7280; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# Column name mapping: raw export headers -> internal names
COLUMN_MAP = {
    "TOTAL D (m)": "total_d",
    "HSR D (m) 19.8 - 25.2 kmh": "hsr_d",
    "SPRINT D (m) 25.2+ kmh": "sprint_d",
    "TOP SPEED (kmh)": "top_speed",
    "ACCEL COUNT 2+ m/s": "accel_count",
    "DECEL COUNT 2+ m/s": "decel_count",
    "DATE": "date",
    "OPPOSITION": "opposition",
    "NAME": "name",
    "TIME": "time",
    "RESULT": "result",
    "FOR": "for_goals",
    "AGAINST": "against_goals",
    "Unit Position": "unit_position",
    "Position": "position",
    "Dist/min": "dist_min",
    "HSR/min": "hsr_min",
    "Sprint D/min": "sprint_d_min",
    "Acc/min": "acc_min",
    "Dec/min": "dec_min",
}

# Editable dictionary of common data-entry corrections
OPPOSITION_CORRECTIONS = {
    "Vanuata": "Vanuatu",
    "Vanuatua": "Vanuatu",
    # add more corrections here as needed
}

NUMERIC_COLS = [
    "total_d", "hsr_d", "sprint_d", "top_speed", "accel_count", "decel_count",
    "dist_min", "hsr_min", "sprint_d_min", "acc_min", "dec_min",
]

STRING_COLS = ["opposition", "name", "unit_position", "position", "result"]


# -----------------------
# Data loading and cleaning (cached together so filters don't retrigger a reload)
# -----------------------
def _parse_minutes(x) -> float:
    """Extract minutes played from a TIME field that may be '90', '90:00', etc."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    try:
        if ":" in s:
            return float(s.split(":")[0])
        return float(s)
    except (ValueError, TypeError):
        return np.nan


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns}, inplace=True)

    for c in STRING_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    if "opposition" in df.columns:
        df["opposition"] = df["opposition"].replace(OPPOSITION_CORRECTIONS)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["minutes"] = df["time"].apply(_parse_minutes) if "time" in df.columns else np.nan

    # Build a readable match label, guarding against missing/NaT dates
    if "date" in df.columns and "opposition" in df.columns:
        date_str = df["date"].dt.strftime("%Y-%m-%d").fillna("Unknown date")
        opp_str = df["opposition"].fillna("Unknown opposition")
        df["match"] = date_str + " vs " + opp_str
    elif "opposition" in df.columns:
        df["match"] = df["opposition"]
    else:
        df["match"] = "Unknown"

    if "for_goals" in df.columns and "against_goals" in df.columns:
        df["score"] = df["for_goals"].fillna("").astype(str) + " - " + df["against_goals"].fillna("").astype(str)
    else:
        df["score"] = ""

    df["outcome"] = df["result"] if "result" in df.columns else ""

    key_fields = [c for c in ["name", "date", "opposition"] if c in df.columns]
    df["data_quality_missing"] = df[key_fields].isnull().any(axis=1) if key_fields else False

    return df


@st.cache_data
def load_and_clean(path: Path) -> pd.DataFrame:
    """Load the CSV and clean it in one cached step, so widget interactions don't re-read disk."""
    raw = pd.read_csv(path)
    return clean_data(raw)


# -----------------------
# Filter application
# -----------------------
def apply_filters(
    df: pd.DataFrame,
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]],
    oppositions: List[str],
    outcomes: List[str],
    players: List[str],
    unit_positions: List[str],
    positions: List[str],
    min_minutes: float,
    minute_threshold: float,
) -> pd.DataFrame:
    d = df.copy()
    if date_range is not None and "date" in d.columns:
        start, end = date_range
        d = d[(d["date"] >= pd.to_datetime(start)) & (d["date"] <= pd.to_datetime(end))]
    if oppositions:
        d = d[d["opposition"].isin(oppositions)]
    if outcomes:
        d = d[d["outcome"].isin(outcomes)]
    if players:
        d = d[d["name"].isin(players)]
    if unit_positions:
        d = d[d["unit_position"].isin(unit_positions)]
    if positions:
        d = d[d["position"].isin(positions)]
    if "minutes" in d.columns and not pd.isna(min_minutes):
        d = d[d["minutes"] >= min_minutes]
    d = d.copy()
    d["minutes_threshold_ok"] = d["minutes"] >= minute_threshold
    return d


# -----------------------
# KPI helpers
# -----------------------
def format_kpi(val, is_int: bool = True, suffix: str = "") -> str:
    if val is None or pd.isna(val):
        return "-"
    if is_int:
        return f"{int(val):,}{suffix}"
    return f"{val:,.1f}{suffix}"


def create_kpi_cards(kpis: Dict[str, Tuple], col_count: int = 4) -> None:
    cols = st.columns(col_count)
    for i, (k, (v, fmt_is_int, suffix)) in enumerate(kpis.items()):
        with cols[i % col_count]:
            st.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-title">{k}</div>'
                f'<div class="kpi-value">{format_kpi(v, fmt_is_int, suffix)}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


def safe_mean(series: Optional[pd.Series]) -> float:
    return series.mean() if series is not None and not series.dropna().empty else np.nan


def safe_max(series: Optional[pd.Series]) -> float:
    return series.max() if series is not None and not series.dropna().empty else np.nan


# -----------------------
# Tabs
# -----------------------
def build_overview_tab(d: pd.DataFrame, metric_for_insight: str) -> None:
    st.header("Overview")
    st.write("Quick season summary and coach-friendly insights.")

    total_matches = d["match"].nunique()
    total_apps = d.shape[0]
    unique_players = d["name"].nunique() if "name" in d.columns else 0

    kpis = {
        "Matches": (total_matches, True, ""),
        "Player appearances": (total_apps, True, ""),
        "Unique players": (unique_players, True, ""),
        "Avg minutes": (safe_mean(d.get("minutes")), False, ""),
        "Avg total distance (m)": (safe_mean(d.get("total_d")), True, ""),
        "Avg HSR distance (m)": (safe_mean(d.get("hsr_d")), True, ""),
        "Avg Sprint distance (m)": (safe_mean(d.get("sprint_d")), True, ""),
        "Max top speed (km/h)": (safe_max(d.get("top_speed")), False, ""),
    }
    create_kpi_cards(kpis, col_count=4)

    st.subheader("Coach Insights")
    with st.expander("Automatic insights (high level)", expanded=True):
        insights = []
        if "total_d" in d.columns and not d["total_d"].dropna().empty:
            row = d.loc[d["total_d"].idxmax()]
            insights.append(
                f"Highest single appearance total distance: {row['total_d']:.0f} m — "
                f"{row.get('name', 'Unknown')} on {row.get('match', '')}."
            )
        if "hsr_min" in d.columns and not d["hsr_min"].dropna().empty:
            avg_hsr_min = d.groupby("name")["hsr_min"].mean()
            top = avg_hsr_min.idxmax()
            insights.append(f"Highest average HSR/min: {avg_hsr_min.max():.2f} — {top} (avg across appearances).")
        if "top_speed" in d.columns and not d["top_speed"].dropna().empty:
            row = d.loc[d["top_speed"].idxmax()]
            insights.append(
                f"Highest top speed observed: {row['top_speed']:.1f} km/h — "
                f"{row.get('name', 'Unknown')} on {row.get('match', '')}."
            )
        if {"sprint_d", "total_d"}.issubset(d.columns):
            sprint_pct = (d["sprint_d"] / d["total_d"]).replace([np.inf, -np.inf], np.nan)
            if not sprint_pct.dropna().empty:
                idx = sprint_pct.idxmax()
                insights.append(
                    f"Largest sprint contribution to match distance: {sprint_pct[idx]:.2%} — "
                    f"{d.loc[idx].get('name', 'Unknown')} ({d.loc[idx].get('match', '')})."
                )

        if insights:
            for it in insights:
                st.write("- " + it)
        else:
            st.info("Not enough data in the current filter selection to generate insights.")

    st.markdown("---")

    st.subheader("Team match profile (aggregated)")
    agg_cols = {c: (c, "sum") for c in ["total_d", "hsr_d", "sprint_d"] if c in d.columns}
    if agg_cols:
        agg_match = d.groupby("match").agg(**agg_cols).reset_index()
    else:
        agg_match = pd.DataFrame()

    if not agg_match.empty and "total_d" in agg_match.columns:
        c1, c2 = st.columns([2, 1])
        with c1:
            fig = px.bar(
                agg_match.sort_values("total_d", ascending=False),
                x="match", y="total_d",
                title="Team total distance by match",
                labels={"total_d": "Total distance (m)"},
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.write("**Interpretation**")
            st.write(
                "Higher team total distance may indicate more running demand in the match — "
                "consider context like opponent, game state, and minutes played."
            )
    else:
        st.info("Not enough data to show match profiles.")


def build_match_demands_tab(d: pd.DataFrame) -> None:
    st.header("Match Demands")
    st.write("Explore match-by-match team demands. Use the aggregation selector to view sum, mean or median.")

    agg_mode = st.selectbox("Aggregate by", ["sum", "mean", "median"])
    metrics = [c for c in ["total_d", "hsr_d", "sprint_d"] if c in d.columns]

    if not metrics:
        st.info("No match-level distance metrics available for current filters.")
        return

    agg = d.groupby("match").agg(**{c: (c, agg_mode) for c in metrics}).reset_index()
    if agg.empty:
        st.info("No match-level data available for current filters.")
        return

    titles = {"total_d": "Total distance (m)", "hsr_d": "HSR distance (m)", "sprint_d": "Sprint distance (m)"}
    for c in metrics:
        fig = px.bar(
            agg.sort_values(c, ascending=False), x="match", y=c,
            title=f"Team {titles[c]} by match ({agg_mode})",
            labels={c: titles[c]},
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    if {"total_d", "hsr_d", "outcome"}.issubset(d.columns):
        fig4 = px.scatter(
            d, x="total_d", y="hsr_d", color="outcome",
            hover_data=["name", "match", "minutes"],
            title="Total distance vs HSR distance (by appearance)",
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown(
        "Coach note: Use these charts to identify matches with unusually high or low running demands; "
        "consider match context (opponent, scoreline, substitutions)."
    )


def build_player_profiles_tab(d: pd.DataFrame) -> None:
    st.header("Player Profiles")

    if "name" not in d.columns or d["name"].dropna().empty:
        st.info("No player names available for the current filters.")
        return

    all_players = sorted(d["name"].dropna().unique())
    player_sel = st.multiselect("Select player(s)", options=all_players, default=all_players[:1])
    if not player_sel:
        st.info("Select at least one player to view profiles.")
        return

    sel_df = d[d["name"].isin(player_sel)].sort_values("date")
    if sel_df.empty:
        st.info("No data for selected player(s) with current filters.")
        return

    metrics = [
        ("total_d", "Total distance (m)"), ("hsr_d", "HSR distance (m)"), ("sprint_d", "Sprint distance (m)"),
        ("top_speed", "Top speed (km/h)"), ("dist_min", "Dist/min"), ("hsr_min", "HSR/min"),
        ("sprint_d_min", "Sprint D/min"),
    ]
    for col, title in metrics:
        if col in sel_df.columns:
            fig = px.line(sel_df, x="date", y=col, color="match", markers=True, title=f"{title} — trend")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Player summary")
    agg_spec = {}
    if "match" in sel_df.columns:
        agg_spec["appearances"] = ("match", "nunique")
    if "minutes" in sel_df.columns:
        agg_spec["avg_minutes"] = ("minutes", "mean")
    for col, label in [("total_d", "avg_total_d"), ("hsr_d", "avg_hsr"), ("sprint_d", "avg_sprint")]:
        if col in sel_df.columns:
            agg_spec[label] = (col, "mean")
    if "total_d" in sel_df.columns:
        agg_spec["max_total_d"] = ("total_d", "max")
    if "top_speed" in sel_df.columns:
        agg_spec["max_top_speed"] = ("top_speed", "max")

    if agg_spec:
        summary = sel_df.groupby("name").agg(**agg_spec).reset_index()
        sort_col = "avg_total_d" if "avg_total_d" in summary.columns else summary.columns[-1]
        st.dataframe(summary.sort_values(sort_col, ascending=False))

    # Radar chart: player vs squad average, normalised 0-100 against the filtered dataset's max
    st.subheader("Normalised profile vs squad average")
    metric_keys = [k for k in ["total_d", "hsr_d", "sprint_d", "top_speed", "dist_min"] if k in d.columns]
    if not metric_keys:
        st.info("No metrics available to build a radar comparison.")
        return

    try:
        max_vals = d[metric_keys].max()
        max_vals = max_vals.where(max_vals > 0, other=1)  # avoid division by zero
        squad_avg = d[metric_keys].mean()
        radar_df = sel_df.groupby("name")[metric_keys].mean().reset_index()

        for _, row in radar_df.iterrows():
            player = row["name"]
            norm_player = [
                min(100, (row[k] / max_vals[k]) * 100) if not pd.isna(row[k]) else 0
                for k in metric_keys
            ]
            norm_squad = [
                min(100, (squad_avg[k] / max_vals[k]) * 100) if not pd.isna(squad_avg[k]) else 0
                for k in metric_keys
            ]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=norm_player, theta=metric_keys, fill="toself", name=player))
            fig.add_trace(go.Scatterpolar(r=norm_squad, theta=metric_keys, fill="toself", name="Squad avg"))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True,
                title=f"Normalized profile — {player}",
            )
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"Radar chart could not be created for the selected players ({e}).")


def build_position_comparison_tab(d: pd.DataFrame) -> None:
    st.header("Position comparison")

    group_options = [c for c in ["unit_position", "position"] if c in d.columns]
    if not group_options:
        st.info("No position fields available in the dataset.")
        return

    mode = st.radio("Group by", group_options, index=0)
    group_col = mode
    st.write(f"Grouping by: {group_col}")

    metrics_box = ["dist_min", "hsr_min", "sprint_d_min", "acc_min", "dec_min"]
    available = [m for m in metrics_box if m in d.columns]
    if not available:
        st.info("No per-minute metrics available to compare.")
        return

    for m in available:
        fig = px.box(d, x=group_col, y=m, points="all", title=f"{m} by {group_col}", labels={m: m, group_col: group_col})
        st.plotly_chart(fig, use_container_width=True)

    agg_spec = {}
    for col, label in [("total_d", "avg_total_d"), ("hsr_d", "avg_hsr"), ("sprint_d", "avg_sprint"), ("top_speed", "avg_top_speed")]:
        if col in d.columns:
            agg_spec[label] = (col, "mean")
    if "name" in d.columns:
        agg_spec["n"] = ("name", "count")

    if agg_spec:
        agg = d.groupby(group_col).agg(**agg_spec).reset_index()
        st.subheader("Average outputs by group (with sample size)")
        sort_col = "avg_total_d" if "avg_total_d" in agg.columns else agg.columns[-1]
        st.dataframe(agg.sort_values(sort_col, ascending=False))

        rankable = [c for c in ["avg_total_d", "avg_hsr", "avg_sprint", "avg_top_speed"] if c in agg.columns]
        if rankable:
            metric_choice = st.selectbox("Rank positions by", rankable)
            st.write("Ranked positions (metric, sample size)")
            ranked = agg.sort_values(metric_choice, ascending=False)[[group_col, metric_choice, "n"]]
            st.dataframe(ranked)


def build_sprint_speed_tab(d: pd.DataFrame) -> None:
    st.header("Sprint and Top Speed")

    if {"hsr_d", "sprint_d", "minutes"}.issubset(d.columns):
        color_col = "unit_position" if "unit_position" in d.columns else ("position" if "position" in d.columns else None)
        fig = px.scatter(
            d, x="hsr_d", y="sprint_d", size="minutes", color=color_col,
            hover_data=["name", "match"], title="HSR vs Sprint distance (size=minutes)",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 10 match outputs — Sprint distance")
    if "sprint_d" in d.columns:
        top10_sprint = d.sort_values("sprint_d", ascending=False).head(10)[["name", "match", "sprint_d", "minutes"]]
        st.dataframe(top10_sprint)

    st.subheader("Top 10 match outputs — Top speed")
    if "top_speed" in d.columns:
        top10_top = d.sort_values("top_speed", ascending=False).head(10)[["name", "match", "top_speed", "minutes"]]
        st.dataframe(top10_top)

    st.markdown("Note: Top speed interpretation should consider minutes played, role, and exposure opportunity.")


def build_accel_decel_tab(d: pd.DataFrame) -> None:
    st.header("Acceleration and Deceleration")

    if {"acc_min", "dec_min"}.issubset(d.columns):
        color_col = "unit_position" if "unit_position" in d.columns else ("position" if "position" in d.columns else None)
        fig = px.scatter(d, x="acc_min", y="dec_min", color=color_col, hover_data=["name", "match"], title="Acc/min vs Dec/min")
        st.plotly_chart(fig, use_container_width=True)

    if "accel_count" in d.columns and "decel_count" in d.columns and "name" in d.columns:
        st.subheader("Accel / Decel counts by player")
        agg = d.groupby("name").agg(
            total_accel=("accel_count", "sum"),
            total_decel=("decel_count", "sum"),
            appearances=("match", "nunique"),
        ).reset_index()
        st.dataframe(agg.sort_values("total_accel", ascending=False).head(50))

    if {"acc_min", "dec_min"}.issubset(d.columns):
        d = d.copy()
        d["accel_decel_balance"] = d["acc_min"] - d["dec_min"]
        if not d["accel_decel_balance"].dropna().empty:
            p75 = d["accel_decel_balance"].quantile(0.75)
            flagged = d[d["accel_decel_balance"] >= p75]
            st.write(f"Players/appearances in upper quartile for accel-decel balance (>= {p75:.2f}) may warrant review (n={flagged.shape[0]})")
            st.dataframe(
                flagged[["name", "match", "acc_min", "dec_min", "accel_decel_balance"]]
                .sort_values("accel_decel_balance", ascending=False)
                .head(50)
            )


def build_data_table_tab(d: pd.DataFrame) -> None:
    st.header("Data Table")
    st.write("Filtered dataset — use this to export and inspect raw values.")
    st.dataframe(d.reset_index(drop=True))

    csv = d.to_csv(index=False)
    st.download_button("Download filtered data as CSV", csv, file_name="filtered_gps_data.csv")

    st.subheader("Summary by player")
    if {"name", "match"}.issubset(d.columns):
        agg_spec = {"appearances": ("match", "count")}
        if "minutes" in d.columns:
            agg_spec["minutes"] = ("minutes", "mean")
        if "total_d" in d.columns:
            agg_spec["total_d"] = ("total_d", "sum")
        summary = d.groupby(["name", "match"]).agg(**agg_spec).reset_index()
        st.dataframe(summary.head(200))


# -----------------------
# Main app
# -----------------------
def main() -> None:
    DATA_PATH = Path(__file__).parent / "data" / "AW_GPS_Data.csv"

    try:
        df = load_and_clean(DATA_PATH)
    except FileNotFoundError:
        st.error(f"CSV file not found at {DATA_PATH}. Place AW_GPS_Data.csv in the data/ folder next to this app.")
        return
    except pd.errors.EmptyDataError:
        st.error("The CSV file is empty. Please check the source export.")
        return
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return

    with st.expander("Data quality checks"):
        st.write("Rows with missing key fields (name, date, opposition):")
        missing = df[df["data_quality_missing"]]
        st.write(f"{missing.shape[0]} rows with missing fields")
        if not missing.empty:
            st.dataframe(missing.head(50))

    key_fields = ["name", "date", "opposition", "total_d"]
    missing_keys = [k for k in key_fields if k not in df.columns]
    if missing_keys:
        st.warning(f"Missing key fields in dataset: {missing_keys} — some features will be disabled.")

    # ---- Sidebar filters ----
    st.sidebar.header("Filters")

    min_date = df["date"].min() if "date" in df.columns else None
    max_date = df["date"].max() if "date" in df.columns else None
    default_min = min_date if pd.notna(min_date) else pd.Timestamp.today()
    default_max = max_date if pd.notna(max_date) else pd.Timestamp.today()

    date_range = st.sidebar.date_input("Date range", value=(default_min, default_max))
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        date_range_tuple = (pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]))
    else:
        single = pd.to_datetime(date_range)
        date_range_tuple = (single, single)

    oppositions = sorted(df["opposition"].dropna().unique()) if "opposition" in df.columns else []
    selected_oppositions = st.sidebar.multiselect("Opposition", oppositions, default=oppositions)

    outcomes = sorted(df["outcome"].dropna().unique()) if "outcome" in df.columns else []
    selected_outcomes = st.sidebar.multiselect("Result", outcomes, default=outcomes)

    players = sorted(df["name"].dropna().unique()) if "name" in df.columns else []
    selected_players = st.sidebar.multiselect("Player", players, default=[])

    unit_positions = sorted(df["unit_position"].dropna().unique()) if "unit_position" in df.columns else []
    selected_unit_positions = st.sidebar.multiselect("Unit Position", unit_positions, default=unit_positions)

    positions = sorted(df["position"].dropna().unique()) if "position" in df.columns else []
    selected_positions = st.sidebar.multiselect("Position", positions, default=positions)

    min_minutes = st.sidebar.slider("Minimum minutes", min_value=0, max_value=120, value=0)
    minute_threshold = st.sidebar.number_input("Minute threshold for inclusion (plotting)", value=10)

    metric_options = [
        ("total_d", "Total distance (m)"), ("hsr_d", "HSR distance (m)"), ("sprint_d", "Sprint distance (m)"),
        ("top_speed", "Top speed (km/h)"), ("dist_min", "Dist/min"), ("hsr_min", "HSR/min"),
        ("sprint_d_min", "Sprint D/min"), ("acc_min", "Acc/min"), ("dec_min", "Dec/min"),
    ]
    available_metrics = [m for m, label in metric_options if m in df.columns]

    if available_metrics:
        selected_metric = st.sidebar.selectbox("Primary metric for insights", options=available_metrics, index=0)
    else:
        selected_metric = None
        st.sidebar.info("No metrics available for insights.")

    st.sidebar.selectbox("Comparison mode", ["Player", "Position", "Match", "Result"])

    st.sidebar.markdown("---")
    st.sidebar.markdown("Minute threshold is used to flag short appearances. Use filters to narrow the dataset.")
    st.sidebar.markdown(
        '<div class="sidebar-note">To reset filters: refresh the page or clear selections in the sidebar.</div>',
        unsafe_allow_html=True,
    )

    filtered = apply_filters(
        df, date_range_tuple, selected_oppositions, selected_outcomes, selected_players,
        selected_unit_positions, selected_positions, min_minutes, minute_threshold,
    )

    if filtered.empty:
        st.warning("No data after applying filters. Adjust sidebar filters.")
        return

    tabs = st.tabs(
        ["Overview", "Match Demands", "Player Profiles", "Position Comparison", "Sprint and Speed", "Accel and Decel", "Data Table"]
    )
    with tabs[0]:
        build_overview_tab(filtered, selected_metric)
    with tabs[1]:
        build_match_demands_tab(filtered)
    with tabs[2]:
        build_player_profiles_tab(filtered)
    with tabs[3]:
        build_position_comparison_tab(filtered)
    with tabs[4]:
        build_sprint_speed_tab(filtered)
    with tabs[5]:
        build_accel_decel_tab(filtered)
    with tabs[6]:
        build_data_table_tab(filtered)


if __name__ == "__main__":
    main()
