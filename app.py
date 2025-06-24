import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# ---- LOAD DATA ----
combine_path = "combine_athletic_metrics_FINAL.csv"
df_combine = pd.read_csv(combine_path)

# List your longform stat files (add more as you want)
stat_files = {
    "College Rushing": "College_Rushing_Summary_Longform_2014_2024.csv",
    "College Receiving": "College_Receiving_Summary_Longform_2014_2024.csv",
    "College Passing": "College_Passing_Summary_Longform_2014_2024.csv",
    "College Defense": "College_Defense_Summary_Longform_2014_2024.csv",
    "College Blocking": "College_Blocking_Longform_2014_2024.csv",
    "NFL Rushing": "NFL_Rushing_Summary_Longform_2010_2024.csv",
    "NFL Receiving": "NFL_Receiving_Summary_Longform_2010_2024.csv",
    "NFL Passing": "NFL_Passing_Summary_Longform_2010_2024.csv",
    "NFL Defense": "NFL_Defense_Summary_Longform_2010_2024.csv",
    "NFL Blocking": "NFL_Blocking_Summary_Longform_2010_2024.csv",
}

# Load all player stats into one dict
player_stats = {}
for label, path in stat_files.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        player_stats[label] = df

# ---- SIDEBAR: Filters ----
st.title("Dynasty Football Combine + Career Explorer")

years = sorted(df_combine['year'].dropna().unique(), reverse=True)
year = st.sidebar.selectbox("Select Combine Year", years)

positions = sorted(df_combine[df_combine['year'] == year]['position'].dropna().unique())
position = st.sidebar.selectbox("Select Position Group", positions)

athletic_cols = ['40yd', 'Bench', 'Vertical', 'Broad Jump', 'Shuttle', '3Cone']

# ---- Player selection (allow two for side-by-side) ----
df_pos = df_combine[(df_combine['year'] == year) & (df_combine['position'] == position)].copy()
metric_availability = df_pos[athletic_cols].notnull().sum(axis=1)
enough_metrics = metric_availability >= 3
df_pos = df_pos[enough_metrics]

player_names = df_pos['player_name'].sort_values().unique()
player1 = st.sidebar.selectbox("Player 1", player_names)
player2 = st.sidebar.selectbox("Player 2 (Compare)", player_names, index=min(1, len(player_names)-1))

# Filter out columns that are all NaN for this sample
non_nan_cols = [c for c in athletic_cols if df_pos[c].notnull().any()]
if len(non_nan_cols) < 2 or len(df_pos) < 2:
    st.warning("Not enough data for PCA in this group.")
    pca = None
    df_pos['PC1'] = np.nan
    df_pos['PC2'] = np.nan
else:
    means = df_pos[non_nan_cols].mean()
    X = df_pos[non_nan_cols].fillna(means)
    X_std = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_std)
    df_pos['PC1'] = components[:, 0]
    df_pos['PC2'] = components[:, 1]


# PCA loadings meaning
pca_loadings = pd.DataFrame(
    pca.components_.T,
    columns=['PC1', 'PC2'],
    index=athletic_cols
)
pc1_top = pca_loadings['PC1'].abs().sort_values(ascending=False).index[0]
pc2_top = pca_loadings['PC2'].abs().sort_values(ascending=False).index[0]
st.markdown(f"""
**PCA Axes:**  
- PC1 is most influenced by: **{pc1_top}**
- PC2 is most influenced by: **{pc2_top}**
""")
st.markdown(f"_PC1 and PC2 summarize how {position}s differ athletically in {year}. High PC1 = high {pc1_top}._")

sel1 = df_pos[df_pos['player_name'] == player1]
sel2 = df_pos[df_pos['player_name'] == player2]

# ---- PCA CHART (show both players) ----
fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(df_pos['PC1'], df_pos['PC2'], alpha=0.5, c='gray', label='All')
if len(sel1):
    ax.scatter(sel1['PC1'], sel1['PC2'], color='red', s=120, edgecolor='black', label=player1)
    for _, row in sel1.iterrows():
        ax.annotate(row['player_name'], (row['PC1'], row['PC2']), fontsize=10, color='red', xytext=(5,5), textcoords='offset points')
if len(sel2):
    ax.scatter(sel2['PC1'], sel2['PC2'], color='blue', s=120, edgecolor='black', label=player2)
    for _, row in sel2.iterrows():
        ax.annotate(row['player_name'], (row['PC1'], row['PC2']), fontsize=10, color='blue', xytext=(-60,-10), textcoords='offset points')
ax.set_xlabel("PC1 (see above)")
ax.set_ylabel("PC2 (see above)")
ax.set_title(f'Combine PCA: {year} {position}s')
ax.grid(True)
ax.legend()
st.pyplot(fig)

# ---- SPIDER CHARTS ----
def plot_spider_chart(percentiles, metrics, label=None, color='red'):
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    percentiles += percentiles[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
    ax.plot(angles, percentiles, color=color, linewidth=2)
    ax.fill(angles, percentiles, color=color, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(['0', '50', '100'])
    if label:
        ax.set_title(label, size=14, y=1.10)
    plt.tight_layout()
    return fig

if len(sel1) and len(sel2):
    cols = st.columns(2)
    for i, (sel, color, label) in enumerate(zip([sel1, sel2], ['red', 'blue'], [player1, player2])):
        with cols[i]:
            st.markdown(f"#### {label} — Spider Chart")
            metrics = [m for m in athletic_cols if sel.iloc[0][m] == sel.iloc[0][m]]
            sel_stats = sel[metrics].iloc[0]
            pos_stats = df_pos[metrics]
            percentiles = [(pos_stats[m] <= sel_stats[m]).mean() for m in metrics]
            fig2 = plot_spider_chart(percentiles, metrics, label, color=color)
            st.pyplot(fig2)
            st.markdown(f"#### {label} — Combine Metrics")
            st.dataframe(sel[athletic_cols + ['PC1', 'PC2']])

# ---- CAREER STATS (if available, e.g. rushing/receiving/passing) ----
def show_career_stats(player, player_stats, st):
    for label, df in player_stats.items():
        if 'player_name' not in df.columns or 'year' not in df.columns:
            continue
        years = sorted(df[df['player_name'] == player]['year'].unique())
        if not years:
            continue
        stats = df[df['player_name'] == player]
        st.markdown(f"#### {player} — {label} (Yearly)")
        stat_cols = [c for c in stats.columns if c not in ['player_name','year','team_name','school','position','stat_type','value']]
        for stat_col in stat_cols:
            # Use only meaningful counting stats for the chart
            stat_pivot = stats.pivot_table(index='year', columns='stat_type', values='value', aggfunc='sum')
            # Pick common stat types
            top_stats = [s for s in ['yards','attempts','touchdowns','carries','receptions','targets','completions'] if s in stat_pivot.columns]
            if not top_stats:
                continue
            st.line_chart(stat_pivot[top_stats])
            break  # Show just one chart per type for now

with st.expander("Show Career Stats (Rushing/Receiving/Passing/Etc.)"):
    show_career_stats(player1, player_stats, st)
    if player2 != player1:
        show_career_stats(player2, player_stats, st)

st.markdown("_Created by Christos' Dynasty Football Project_")

