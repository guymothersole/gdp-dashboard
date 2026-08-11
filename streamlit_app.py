import streamlit as st
import pandas as pd
import math
from pathlib import Path

st.set_page_config(
    page_title='AW GPS dashboard',
    page_icon=':soccer:',
)

@st.cache_data
def load_data():
    DATA_FILENAME = Path(__file__).parent / 'data' / 'AW_GPS_Data.csv'
    df = pd.read_csv(DATA_FILENAME)

    # Clean column names (remove leading/trailing spaces)
    df.columns = df.columns.str.strip()

    # Parse DATE column (samples look like DD/MM/YYYY)
    if 'DATE' in df.columns:
        df['DATE'] = pd.to_datetime(df['DATE'], dayfirst=True, errors='coerce')

    # Ensure numeric columns are numeric
    numeric_cols = [
        'TOTAL D (m)',
        'HSR D (m) 19.8 - 25.2 kmh',
        'SPRINT D (m) 25.2+ kmh',
        'TOP SPEED (kmh)',
        'ACCEL COUNT 2+ m/s',
        'DECEL COUNT 2+ m/s',
    ]

    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Also normalize string columns
    for c in ['OPPOSITION', 'NAME', 'Unit Position']:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    return df


df = load_data()

st.title('AW GPS interactive dashboard')
st.markdown('Filter the dataset by Opposition, Name and Unit Position; then explore the GPS metrics.')

# Guard for missing dataset
if df is None or df.shape[0] == 0:
    st.warning('No data loaded from data/AW_GPS_Data.csv')
    st.stop()

# Sidebar filters
st.sidebar.header('Filters')

oppositions = sorted(df['OPPOSITION'].dropna().unique()) if 'OPPOSITION' in df.columns else []
selected_opposition = st.sidebar.multiselect('Opposition', oppositions, default=oppositions)

# Filter names based on opposition selection
if 'NAME' in df.columns:
    names = df[df['OPPOSITION'].isin(selected_opposition)]['NAME'].dropna().unique() if selected_opposition else df['NAME'].dropna().unique()
    names = sorted(names)
else:
    names = []
selected_names = st.sidebar.multiselect('Name', names, default=list(names)[:5])

unit_positions = sorted(df['Unit Position'].dropna().unique()) if 'Unit Position' in df.columns else []
selected_units = st.sidebar.multiselect('Unit Position', unit_positions, default=unit_positions)

# Filter dataframe
filtered = df.copy()
if selected_opposition:
    filtered = filtered[filtered['OPPOSITION'].isin(selected_opposition)]
if selected_names:
    filtered = filtered[filtered['NAME'].isin(selected_names)]
if selected_units:
    filtered = filtered[filtered['Unit Position'].isin(selected_units)]

if filtered.empty:
    st.warning('No rows match your filter selection.')
    st.stop()

# Metrics to show
metrics = [
    'TOTAL D (m)',
    'HSR D (m) 19.8 - 25.2 kmh',
    'SPRINT D (m) 25.2+ kmh',
    'TOP SPEED (kmh)',
    'ACCEL COUNT 2+ m/s',
    'DECEL COUNT 2+ m/s',
]
available_metrics = [m for m in metrics if m in filtered.columns]
selected_metrics = st.sidebar.multiselect('Metrics to plot', available_metrics, default=available_metrics[:3])

plot_mode = st.sidebar.selectbox('Plot mode', ['Time series (by DATE)', 'Per-player aggregate (bar)'])

st.header('Filtered data')
st.dataframe(filtered.head(200))

if not selected_metrics:
    st.info('Choose one or more metrics to plot from the sidebar.')
    st.stop()

# Plotting
if plot_mode == 'Time series (by DATE)':
    if 'DATE' not in filtered.columns:
        st.error('DATE column not found; cannot create time series.')
    else:
        # For each metric, create a line chart across time. Color by NAME if multiple names selected.
        # We'll pivot so each NAME becomes a column (for a given metric)
        for metric in selected_metrics:
            st.subheader(metric)
            chart_df = filtered[['DATE', 'NAME', metric]].dropna()
            if chart_df.empty:
                st.write('No data for this metric after filtering.')
                continue

            if len(chart_df['NAME'].unique()) <= 1:
                # single player or unspecified: aggregate by DATE
                series = chart_df.groupby('DATE')[metric].sum().sort_index()
                st.line_chart(series)
            else:
                pivot = chart_df.pivot_table(index='DATE', columns='NAME', values=metric, aggfunc='sum')
                st.line_chart(pivot.sort_index())

else:
    st.subheader('Per-player aggregate')
    # Aggregate selected metrics per player (sum) and show as bar charts
    agg = filtered.groupby('NAME')[selected_metrics].sum().sort_values(by=selected_metrics[0], ascending=False)
    st.dataframe(agg)

    # For each metric show a bar chart
    for metric in selected_metrics:
        st.markdown(f'**{metric}**')
        st.bar_chart(agg[metric])

st.sidebar.markdown('---')
st.sidebar.markdown('Tip: use the filters to narrow the data and switch plot mode to inspect per-player totals or time trends.')
