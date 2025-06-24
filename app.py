import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---- LOAD DATA ----
DATA_PATH = "combine_athletic_metrics_FINAL.csv"
df = pd.read_csv(DATA_PATH)

# ---- APP TITLE ----
st.title("Combine PCA Explorer")

# ---- SIDEBAR FOR USER SELECTION ----
positions = sorted(df['position'].dropna().unique())
position = st.sidebar.selectbox("Select Position Group", positions)

# Athletic columns to use (edit as needed)
athletic_cols = ['40yd', 'Bench', 'Vertical', 'Broad Jump', 'Shuttle', '3Cone']

# Only keep rows with enough data for this position
df_pos = df[df['position'] == position].copy()
metric_availability = df_pos[athletic_cols].notnull().sum(axis=1)
enough_metrics = metric_availability >= 3   # Allow players with at least 3 combine metrics

df_pos = df_pos[enough_metrics]

# Standardize with imputation (mean fill)
means = df_pos[athletic_cols].mean()
X = df_pos[athletic_cols].fillna(means)
X_std = StandardScaler().fit_transform(X)

# PCA
pca = PCA(n_components=2)
components = pca.fit_transform(X_std)
df_pos['PC1'] = components[:, 0]
df_pos['PC2'] = components[:, 1]

# ---- PLAYER SELECT ----
player_names = df_pos['player_name'].sort_values().unique()
selected_player = st.sidebar.selectbox("Highlight Player", player_names)

# ---- EXPLAIN PCA COMPONENTS ----
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

st.markdown(f"""
_PC1 and PC2 are synthetic axes that summarize the major ways players differ in their combine results.  
A high PC1 often means high/low in {pc1_top}; PC2 is driven by {pc2_top} for this group._
""")

# ---- PCA PLOT ----
fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(df_pos['PC1'], df_pos['PC2'], alpha=0.6, c='gray', label='All players')

sel = df_pos[df_pos['player_name'] == selected_player]
ax.scatter(sel['PC1'], sel['PC2'], color='red', s=100, edgecolor='black', label=selected_player)

for _, row in sel.iterrows():
    ax.annotate(row['player_name'], (row['PC1'], row['PC2']), fontsize=10, color='red', xytext=(5,5), textcoords='offset points')

ax.set_xlabel("PC1 (see sidebar)")
ax.set_ylabel("PC2 (see sidebar)")
ax.set_title(f'Combine PCA for {position}s')
ax.grid(True)
ax.legend()
st.pyplot(fig)

# ---- PLAYER METRICS ----
if len(sel):
    st.markdown(f"### {selected_player} - Combine Metrics")
    st.dataframe(sel[athletic_cols + ['PC1', 'PC2']])

st.markdown("_Created by Christos' Dynasty Football Project_")
