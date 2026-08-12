import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import List, Dict, Tuple

# Page config
st.set_page_config(page_title='AW GPS — Coach Dashboard', page_icon=':soccer:', layout='wide')

# --- Styles (simple CSS for modern cards) ---
st.markdown(
    """
    <style>
    .kpi-card {
        background: linear-gradient(180deg, #ffffff, #f7f9fc);
        border-radius: 10px;
        padding: 12px;
        box-shadow: 0 1px 3px rgba(16,24,40,0.06);
    }
    .kpi-title { color: #1f2937; font-size:14px; }
    .kpi-value { color: #0f172a; font-weight:700; font-size:20px; }
    .insight { background:#0f172a; color:#fff; padding:10px; border-radius:8px }
    .sidebar-note { font-size:12px; color:#6b7280 }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------
# Data loading and cleaning
# -----------------------

@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Strip column names
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Standardise important column names mapping to simpler internal names (if present)
    col_map = {
        'TOTAL D (m)': 'total_d',
        'HSR D (m) 19.8 - 25.2 kmh': 'hsr_d',
        'SPRINT D (m) 25.2+ kmh': 'sprint_d',
        'TOP SPEED (kmh)': 'top_speed',
        'ACCEL COUNT 2+ m/s': 'accel_count',
        'DECEL COUNT 2+ m/s': 'decel_count',
        'DATE': 'date',
        'OPPOSITION': 'opposition',
        'NAME': 'name',
        'TIME': 'time',
        'RESULT': 'result',
        'FOR': 'for_goals',
        'AGAINST': 'against_goals',
        'Unit Position': 'unit_position',
        'Position': 'position',
        'Dist/min': 'dist_min',
        'HSR/min': 'hsr_min',
        'Sprint D/min': 'sprint_d_min',
        'Acc/min': 'acc_min',
        'Dec/min': 'dec_min',
    }
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    # Trim whitespace from string columns
    for c in ['opposition', 'name', 'unit_position', 'position', 'result']:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # Fix common spelling mistakes in opposition (editable dictionary)
    opposition_cleaning = {
        'Vanuata': 'Vanuatu',
        'Vanuatua': 'Vanuatu',
        # add more corrections as needed
    }
    if 'opposition' in df.columns:
        df['opposition'] = df['opposition'].replace(opposition_cleaning)

    # Parse dates (dayfirst)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')

    # Numeric conversions
    numeric_cols = [
        'total_d', 'hsr_d', 'sprint_d', 'top_speed', 'accel_count', 'decel_count',
        'dist_min', 'hsr_min', 'sprint_d_min', 'acc_min', 'dec_min'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # TIME/minutes handling: try to infer minutes from TIME or ensure numeric 'time' column
    if 'time' in df.columns:
        # If TIME is a string like '90' or '90:00' try to extract minutes
        def parse_minutes(x):
            try:
                if pd.isna(x):
                    return np.nan
                s = str(x).strip()
                if ':' in s:
                    parts = s.split(':')
                    return int(parts[0])
                return float(s)
            except Exception:
                return np.nan
        df['minutes'] = df['time'].apply(parse_minutes)
    else:
        df['minutes'] = np.nan

    # Create match label and score label
    if 'date' in df.columns and 'opposition' in df.columns:
        df['match'] = df['date'].dt.strftime('%Y-%m-%d') + ' vs ' + df['opposition']
    elif 'opposition' in df.columns:
        df['match'] = df['opposition']
    else:
        df['match'] = 'Unknown'

    if 'for_goals' in df.columns and 'against_goals' in df.columns:
        df['score'] = df['for_goals'].astype(str) + ' - ' + df['against_goals'].astype(str)
    else:
        df['score'] = ''

    if 'result' in df.columns:
        df['outcome'] = df['result']
    else:
        df['outcome'] = ''

    # Basic quality: count missing key fields
    df['data_quality_missing'] = df[['name', 'date', 'opposition']].isnull().any(axis=1)

    return df


# -----------------------
# Filter application
# -----------------------

def apply_filters(df: pd.DataFrame,
                  date_range: Tuple[pd.Timestamp, pd.Timestamp],
                  oppositions: List[str],
                  outcomes: List[str],
                  players: List[str],
                  unit_positions: List[str],
                  positions: List[str],
                  min_minutes: float,
                  minute_threshold: float) -> pd.DataFrame:
    d = df.copy()
    if date_range is not None and 'date' in d.columns:
        start, end = date_range
        d = d[(d['date'] >= pd.to_datetime(start)) & (d['date'] <= pd.to_datetime(end))]
    if oppositions:
        d = d[d['opposition'].isin(oppositions)]
    if outcomes:
        d = d[d['outcome'].isin(outcomes)]
    if players:
        d = d[d['name'].isin(players)]
    if unit_positions:
        d = d[d['unit_position'].isin(unit_positions)]
    if positions:
        d = d[d['position'].isin(positions)]
    if 'minutes' in d.columns and not pd.isna(min_minutes):
        d = d[d['minutes'] >= min_minutes]
    # filter minute threshold for plotting decisions (kept as column)
    d = d.copy()
    d['minutes_threshold_ok'] = d['minutes'] >= minute_threshold
    return d


# -----------------------
# KPI helpers
# -----------------------

def format_kpi(val, is_int=True, suffix=''):
    if pd.isna(val):
        return '-'
    if is_int:
        return f"{int(val):,}{suffix}"
    else:
        return f"{val:,.1f}{suffix}"


def create_kpi_cards(kpis: Dict[str, Tuple], col_count: int = 4):
    cols = st.columns(col_count)
    i = 0
    for k, (v, fmt_is_int, suffix) in kpis.items():
        with cols[i % col_count]:
            st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-title">{k}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-value">{format_kpi(v, fmt_is_int, suffix)}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        i += 1


# -----------------------
# Build tabs content
# -----------------------

def build_overview_tab(d: pd.DataFrame, metric_for_insight: str):
    st.header('Overview')
    st.write('Quick season summary and coach-friendly insights.')

    # KPIs
    total_matches = d['match'].nunique()
    total_apps = d.shape[0]
    unique_players = d['name'].nunique() if 'name' in d.columns else 0
    avg_minutes = d['minutes'].mean()
    avg_total_d = d['total_d'].mean() if 'total_d' in d.columns else np.nan
    avg_hsr = d['hsr_d'].mean() if 'hsr_d' in d.columns else np.nan
    avg_sprint = d['sprint_d'].mean() if 'sprint_d' in d.columns else np.nan
    max_top_speed = d['top_speed'].max() if 'top_speed' in d.columns else np.nan

    kpis = {
        'Matches': (total_matches, True, ''),
        'Player appearances': (total_apps, True, ''),
        'Unique players': (unique_players, True, ''),
        'Avg minutes': (round(avg_minutes or 0, 1), False, ''),
        'Avg total distance (m)': (round(avg_total_d or 0, 0), True, ''),
        'Avg HSR distance (m)': (round(avg_hsr or 0, 0), True, ''),
        'Avg Sprint distance (m)': (round(avg_sprint or 0, 0), True, ''),
        'Max top speed (km/h)': (round(max_top_speed or 0, 1), False, ''),
    }
    create_kpi_cards(kpis, col_count=4)

    # Insight summary
    st.subheader('Coach Insights')
    with st.expander('Automatic insights (high level)'):
        insights = []
        # Highest match total distance appearance
        if 'total_d' in d.columns and not d['total_d'].dropna().empty:
            row = d.loc[d['total_d'].idxmax()]
            insights.append(f"Highest single appearance total distance: {row['total_d']:.0f} m — {row.get('name','Unknown')} on {row.get('match','')}.")
        # Highest average HSR per minute
        if 'hsr_min' in d.columns and not d['hsr_min'].dropna().empty:
            avg_hsr_min = d.groupby('name')['hsr_min'].mean()
            top = avg_hsr_min.idxmax()
            insights.append(f"Highest average HSR/min: {avg_hsr_min.max():.2f} — {top} (avg across appearances).")
        # Highest top speed
        if 'top_speed' in d.columns and not d['top_speed'].dropna().empty:
            row = d.loc[d['top_speed'].idxmax()]
            insights.append(f"Highest top speed observed: {row['top_speed']:.1f} km/h — {row.get('name','Unknown')} on {row.get('match','')}.")
        # Sprint contribution
        if 'sprint_d' in d.columns and 'total_d' in d.columns:
            d = d.assign(sprint_pct = (d['sprint_d'] / d['total_d']).replace([np.inf, -np.inf], np.nan))
            s = d.loc[d['sprint_pct'].idxmax()]
            insights.append(f"Largest sprint contribution to match distance: {s['sprint_pct']:.2%} — {s.get('name','Unknown')} ({s.get('match','')}).")

        for it in insights:
            st.write('- ' + it)

    st.markdown('---')

    # Small charts: average profiles by match (team aggregated)
    st.subheader('Team match profile (aggregated)')
    agg_match = d.groupby('match').agg(
        total_d=('total_d', 'sum'), hsr_d=('hsr_d','sum'), sprint_d=('sprint_d','sum')
    ).reset_index()
    if not agg_match.empty:
        c1, c2 = st.columns([2,1])
        with c1:
            fig = px.bar(agg_match.sort_values('total_d', ascending=False), x='match', y='total_d', title='Team total distance by match', labels={'total_d':'Total distance (m)'})
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.write('Interpretation')
            st.write('Higher team total distance may indicate more running demand in the match — consider context like opponent, game state, and minutes played.')
    else:
        st.info('Not enough data to show match profiles.')


def build_match_demands_tab(d: pd.DataFrame):
    st.header('Match Demands')
    st.write('Explore match-by-match team demands. Use aggregation selector to view sum, mean or median.')

    agg_mode = st.selectbox('Aggregate by', ['sum', 'mean', 'median'])

    agg_funcs = {'sum':'sum', 'mean':'mean', 'median':'median'}
    agg = d.groupby('match').agg(
        total_d=('total_d', agg_funcs[agg_mode] if 'total_d' in d.columns else 'sum'),
        hsr_d=('hsr_d', agg_funcs[agg_mode] if 'hsr_d' in d.columns else 'sum'),
        sprint_d=('sprint_d', agg_funcs[agg_mode] if 'sprint_d' in d.columns else 'sum'),
    ).reset_index()

    if agg.empty:
        st.info('No match-level data available for current filters.')
        return

    fig1 = px.bar(agg.sort_values('total_d', ascending=False), x='match', y='total_d', title=f'Team total distance by match ({agg_mode})', labels={'total_d':'Total distance (m)'} )
    fig1.update_layout(xaxis_tickangle=-45)
    fig2 = px.bar(agg.sort_values('hsr_d', ascending=False), x='match', y='hsr_d', title=f'Team HSR distance by match ({agg_mode})', labels={'hsr_d':'HSR distance (m)'} )
    fig2.update_layout(xaxis_tickangle=-45)
    fig3 = px.bar(agg.sort_values('sprint_d', ascending=False), x='match', y='sprint_d', title=f'Team Sprint distance by match ({agg_mode})', labels={'sprint_d':'Sprint distance (m)'} )
    fig3.update_layout(xaxis_tickangle=-45)

    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.plotly_chart(fig3, use_container_width=True)

    # Scatter total vs hsr coloured by outcome
    if {'total_d','hsr_d','outcome'}.issubset(d.columns):
        fig4 = px.scatter(d, x='total_d', y='hsr_d', color='outcome', hover_data=['name','match','minutes'], title='Total distance vs HSR distance (by appearance)')
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown('Coach note: Use these charts to identify matches with unusually high or low running demands; consider match context (opponent, scoreline, substitutions).')


def build_player_profiles_tab(d: pd.DataFrame):
    st.header('Player Profiles')
    player_sel = st.multiselect('Select player(s)', options=sorted(d['name'].dropna().unique()), default=sorted(d['name'].dropna().unique())[:1])
    if not player_sel:
        st.info('Select at least one player to view profiles.')
        return

    sel_df = d[d['name'].isin(player_sel)].sort_values('date')
    if sel_df.empty:
        st.info('No data for selected player(s) with current filters.')
        return

    # Trend charts for selected metrics
    metrics = [('total_d','Total distance (m)'), ('hsr_d','HSR distance (m)'), ('sprint_d','Sprint distance (m)'), ('top_speed','Top speed (km/h)'), ('dist_min','Dist/min'), ('hsr_min','HSR/min'), ('sprint_d_min','Sprint D/min')]
    for col, title in metrics:
        if col in sel_df.columns:
            fig = px.line(sel_df, x='date', y=col, color='match', markers=True, title=f'{title} — trend')
            st.plotly_chart(fig, use_container_width=True)

    # Summary table
    st.subheader('Player summary')
    summary = sel_df.groupby('name').agg(
        appearances=('match','nunique'),
        avg_minutes=('minutes','mean'),
        avg_total_d=('total_d','mean'),
        max_total_d=('total_d','max'),
        avg_hsr=('hsr_d','mean'),
        avg_sprint=('sprint_d','mean'),
        max_top_speed=('top_speed','max')
    ).reset_index()
    st.dataframe(summary.sort_values('avg_total_d', ascending=False))

    # Radar chart comparing player average to squad average
    try:
        metric_keys = ['total_d','hsr_d','sprint_d','top_speed','dist_min']
        radar_df = sel_df.groupby('name')[metric_keys].mean().reset_index()
        squad_avg = d.groupby('name')[metric_keys].mean().mean() if not d.empty else None
        for _, row in radar_df.iterrows():
            player = row['name']
            values = [row[k] if not pd.isna(row[k]) else 0 for k in metric_keys]
            # Normalise 0-100 based on filtered dataset
            max_vals = pd.concat([d[metric_keys].max(), pd.Series([1]*len(metric_keys), index=metric_keys)], axis=1).iloc[0]
            norm_player = [min(100, (v / max_vals[k]) * 100 if max_vals[k] > 0 else 0) for k,v,k in zip(values, values, metric_keys)]
            squad_vals = [d[k].mean() if k in d.columns else 0 for k in metric_keys]
            norm_squad = [min(100, (sv / max_vals[k]) * 100 if max_vals[k] > 0 else 0) for sv,k in zip(squad_vals, metric_keys)]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=norm_player, theta=[k for k in metric_keys], fill='toself', name=player))
            fig.add_trace(go.Scatterpolar(r=norm_squad, theta=[k for k in metric_keys], fill='toself', name='Squad avg'))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])), showlegend=True, title=f'Normalized profile — {player}')
            st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.info('Radar chart could not be created for the selected players.')


def build_position_comparison_tab(d: pd.DataFrame):
    st.header('Position comparison')
    mode = st.radio('Group by', ['unit_position','position'], index=0)
    group_col = mode if mode in d.columns else ('unit_position' if 'unit_position' in d.columns else 'position')

    st.write(f'Grouping by: {group_col}')

    metrics_box = ['dist_min','hsr_min','sprint_d_min','acc_min','dec_min']
    available = [m for m in metrics_box if m in d.columns]
    if not available:
        st.info('No per-minute metrics available to compare.')
        return

    # Box plots
    for m in available:
        fig = px.box(d, x=group_col, y=m, points='all', title=f'{m} by {group_col}', labels={m:m, group_col:group_col})
        st.plotly_chart(fig, use_container_width=True)

    # Avg bar charts
    agg = d.groupby(group_col).agg(
        avg_total_d=('total_d','mean'), avg_hsr=('hsr_d','mean'), avg_sprint=('sprint_d','mean'), avg_top_speed=('top_speed','mean'), n=('name','count')
    ).reset_index()
    st.subheader('Average outputs by group (with sample size)')
    st.dataframe(agg.sort_values('avg_total_d', ascending=False))

    # Insight table ranking positions by selected metric
    metric_choice = st.selectbox('Rank positions by', ['avg_total_d','avg_hsr','avg_sprint','avg_top_speed'])
    if metric_choice in agg.columns:
        st.write('Ranked positions (metric, sample size)')
        ranked = agg.sort_values(metric_choice, ascending=False)[[group_col, metric_choice, 'n']]
        st.dataframe(ranked)


def build_sprint_speed_tab(d: pd.DataFrame):
    st.header('Sprint and Top Speed')
    if {'hsr_d','sprint_d','minutes'}.issubset(d.columns):
        fig = px.scatter(d, x='hsr_d', y='sprint_d', size='minutes', color='unit_position' if 'unit_position' in d.columns else 'position', hover_data=['name','match'], title='HSR vs Sprint distance (size=minutes)')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader('Top 10 match outputs — Sprint distance')
    if 'sprint_d' in d.columns:
        top10_sprint = d.sort_values('sprint_d', ascending=False).head(10)[['name','match','sprint_d','minutes']]
        st.dataframe(top10_sprint)

    st.subheader('Top 10 match outputs — Top speed')
    if 'top_speed' in d.columns:
        top10_top = d.sort_values('top_speed', ascending=False).head(10)[['name','match','top_speed','minutes']]
        st.dataframe(top10_top)

    st.markdown('Note: Top speed interpretation should consider minutes played, role, and exposure opportunity.')


def build_accel_decel_tab(d: pd.DataFrame):
    st.header('Acceleration and Deceleration')
    if {'acc_min','dec_min'}.issubset(d.columns):
        fig = px.scatter(d, x='acc_min', y='dec_min', color='unit_position' if 'unit_position' in d.columns else 'position', hover_data=['name','match'], title='Acc/min vs Dec/min')
        st.plotly_chart(fig, use_container_width=True)

    if 'accel_count' in d.columns:
        st.subheader('Accel / Decel counts by player')
        agg = d.groupby('name').agg(total_accel=('accel_count','sum'), total_decel=('decel_count','sum'), appearances=('match','nunique')).reset_index()
        st.dataframe(agg.sort_values('total_accel', ascending=False).head(50))

    # Accel-decel balance
    if {'acc_min','dec_min'}.issubset(d.columns):
        d = d.copy()
        d['accel_decel_balance'] = d['acc_min'] - d['dec_min']
        p75 = d['accel_decel_balance'].quantile(0.75)
        p25 = d['accel_decel_balance'].quantile(0.25)
        st.write(f'Players/appearances in upper quartile for accel-decel balance (>= {p75:.2f}) may warrant review (n={d[d["accel_decel_balance"]>=p75].shape[0]})')
        st.dataframe(d[d['accel_decel_balance']>=p75][['name','match','acc_min','dec_min','accel_decel_balance']].sort_values('accel_decel_balance', ascending=False).head(50))


def build_data_table_tab(d: pd.DataFrame):
    st.header('Data Table')
    st.write('Filtered dataset — use this to export and inspect raw values.')
    st.dataframe(d.reset_index(drop=True))

    # Download
    csv = d.to_csv(index=False)
    st.download_button('Download filtered data as CSV', csv, file_name='filtered_gps_data.csv')

    # Grouped summary example
    st.subheader('Summary by player')
    if 'name' in d.columns:
        summary = d.groupby(['name','match']).agg(appearances=('match','count'), minutes=('minutes','mean'), total_d=('total_d','sum')).reset_index()
        st.dataframe(summary.head(200))


# -----------------------
# Main app
# -----------------------

def main():
    DATA_PATH = Path(__file__).parent / 'data' / 'AW_GPS_Data.csv'

    # Load
    try:
        raw = load_data(DATA_PATH)
    except FileNotFoundError:
        st.error(f'CSV file not found at {DATA_PATH}. Place AW_GPS_Data.csv in the data/ folder next to this app.')
        return

    df = clean_data(raw)

    # Data quality section
    with st.expander('Data quality checks'):
        st.write('Rows with missing key fields (name, date, opposition):')
        missing = df[df['data_quality_missing']]
        st.write(f'{missing.shape[0]} rows with missing fields')
        if not missing.empty:
            st.dataframe(missing.head(50))

    # Warn if key fields missing entirely
    key_fields = ['name','date','opposition','total_d']
    missing_keys = [k for k in key_fields if k not in df.columns]
    if missing_keys:
        st.warning(f'Missing key fields in dataset: {missing_keys} — some features will be disabled.')

    # Sidebar filters
    st.sidebar.header('Filters')
    min_date = df['date'].min() if 'date' in df.columns else None
    max_date = df['date'].max() if 'date' in df.columns else None
    date_range = st.sidebar.date_input('Date range', value=(min_date or pd.Timestamp.today(), max_date or pd.Timestamp.today()))
    # date_input returns list/tuple
    if isinstance(date_range, list) or isinstance(date_range, tuple):
        date_range_tuple = (pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]))
    else:
        date_range_tuple = (pd.to_datetime(date_range), pd.to_datetime(date_range))

    oppositions = sorted(df['opposition'].dropna().unique()) if 'opposition' in df.columns else []
    selected_oppositions = st.sidebar.multiselect('Opposition', oppositions, default=oppositions)

    outcomes = sorted(df['outcome'].dropna().unique()) if 'outcome' in df.columns else []
    selected_outcomes = st.sidebar.multiselect('Result', outcomes, default=outcomes)

    players = sorted(df['name'].dropna().unique()) if 'name' in df.columns else []
    selected_players = st.sidebar.multiselect('Player', players, default=[])

    unit_positions = sorted(df['unit_position'].dropna().unique()) if 'unit_position' in df.columns else []
    selected_unit_positions = st.sidebar.multiselect('Unit Position', unit_positions, default=unit_positions)

    positions = sorted(df['position'].dropna().unique()) if 'position' in df.columns else []
    selected_positions = st.sidebar.multiselect('Position', positions, default=positions)

    min_minutes = st.sidebar.slider('Minimum minutes', min_value=0, max_value=120, value=0)
    minute_threshold = st.sidebar.number_input('Minute threshold for inclusion (plotting)', value=10)

    metric_options = [
        ('total_d','Total distance (m)'), ('hsr_d','HSR distance (m)'), ('sprint_d','Sprint distance (m)'), ('top_speed','Top speed (km/h)'),
        ('dist_min','Dist/min'), ('hsr_min','HSR/min'), ('sprint_d_min','Sprint D/min'), ('acc_min','Acc/min'), ('dec_min','Dec/min')
    ]
    available_metrics = [m for m, label in metric_options if m in df.columns]
    metric_labels = {m:label for m,label in metric_options}
    selected_metric = st.sidebar.selectbox('Primary metric for insights', options=available_metrics, index=0 if available_metrics else None)

    comparison_mode = st.sidebar.selectbox('Comparison mode', ['Player','Position','Match','Result'])

    st.sidebar.markdown('---')
    st.sidebar.markdown('Minute threshold is used to flag short appearances. Use filters to narrow the dataset.')
    st.sidebar.markdown('<div class="sidebar-note">To reset filters: refresh the page or clear selections in the sidebar.</div>', unsafe_allow_html=True)

    # Apply filters
    filtered = apply_filters(df, date_range_tuple, selected_oppositions, selected_outcomes, selected_players, selected_unit_positions, selected_positions, min_minutes, minute_threshold)

    if filtered.empty:
        st.warning('No data after applying filters. Adjust sidebar filters.')
        return

    # Tabs
    tabs = st.tabs(['Overview','Match Demands','Player Profiles','Position Comparison','Sprint and Speed','Accel and Decel','Data Table'])
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


if __name__ == '__main__':
    main()
