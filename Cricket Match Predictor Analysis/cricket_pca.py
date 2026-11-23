"""
PCA on synthetic cricket ball-by-ball data
Follows the workflow in Principal_Component_Analysis.pdf:
  load data -> scale -> PCA(n_components=2) -> scatter plot -> interpret components (heatmap)
Part A: ball-level PCA (each row = one delivery, coloured by wicket / no wicket)
Part B: batter-level PCA (each row = one batter, aggregated stats)
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

CSV_PATH = "synthetic_cricket_ball_by_ball_10k.csv"   # change if needed

# ------------------------------------------------------------------ 1. Data
# keep_default_na=False so the text "None" (no wicket) is not read as NaN
df = pd.read_csv(CSV_PATH, keep_default_na=False)
print(df.shape)
print(df.head(3))

df["is_wicket"] = (df["wicket_type"] != "None").astype(int)   # target, like cancer['target']
df["is_boundary"] = df["runs_off_bat"].isin([4, 6]).astype(int)

# =================================================================
# PART A: Ball-level PCA
# =================================================================
num_cols = ["innings", "over", "ball", "runs_off_bat", "extras", "total_runs"]
cat_cols = ["match_phase", "pitch_type", "weather", "venue"]

# One-hot encode categoricals so PCA gets numeric features only
X = pd.concat([df[num_cols], pd.get_dummies(df[cat_cols], dtype=int)], axis=1)
print("Feature matrix:", X.shape)

# Scale -> unit variance (PCA is scale-sensitive, as in the PDF)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(X)

# PCA with 2 components
pca = PCA(n_components=2)
x_pca = pca.fit_transform(scaled_data)
print(scaled_data.shape, "->", x_pca.shape)
print("Explained variance ratio:", pca.explained_variance_ratio_.round(4))

# Scatter plot coloured by wicket
plt.figure(figsize=(8, 6))
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=df["is_wicket"], cmap="viridis", alpha=0.5, s=12)
plt.xlabel("First principal component")
plt.ylabel("Second Principal Component")
plt.title("Ball-level PCA (yellow = wicket)")
plt.tight_layout()
plt.savefig("pca_ball_scatter.png", dpi=150)
plt.show()

# Interpreting components: heatmap of loadings
df_comp = pd.DataFrame(pca.components_, columns=X.columns, index=["PC1", "PC2"])
plt.figure(figsize=(16, 4))
sns.heatmap(df_comp, cmap="plasma", annot=True, fmt=".2f", annot_kws={"rotation": 90})
plt.title("Ball-level component loadings")
plt.tight_layout()
plt.savefig("pca_ball_heatmap.png", dpi=150)
plt.show()

# Variance explained by ALL components (helps choose n_components)
pca_full = PCA().fit(scaled_data)
cum = np.cumsum(pca_full.explained_variance_ratio_)
plt.figure(figsize=(8, 4))
plt.plot(range(1, len(cum) + 1), cum, marker="o")
plt.axhline(0.9, color="red", ls="--", label="90%")
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.legend()
plt.tight_layout()
plt.savefig("pca_ball_variance.png", dpi=150)
plt.show()
print("Components needed for 90% variance:", int(np.argmax(cum >= 0.9)) + 1)

# =================================================================
# PART B: Batter-level PCA (20 batters, aggregated features)
# =================================================================
g = df.groupby("striker")
bat = pd.DataFrame({
    "balls_faced":      g.size(),
    "runs":             g["runs_off_bat"].sum(),
    "strike_rate":      g["runs_off_bat"].mean() * 100,
    "avg_total_runs":   g["total_runs"].mean(),
    "avg_extras":       g["extras"].mean(),
    "dot_ball_pct":     g["runs_off_bat"].apply(lambda s: (s == 0).mean() * 100),
    "boundary_pct":     g["is_boundary"].mean() * 100,
    "wicket_pct":       g["is_wicket"].mean() * 100,
})
# phase-wise strike rates
phase_sr = (df.pivot_table(index="striker", columns="match_phase",
                           values="runs_off_bat", aggfunc="mean") * 100)
phase_sr.columns = [f"SR_{c}" for c in phase_sr.columns]
bat = bat.join(phase_sr)

bat_scaled = StandardScaler().fit_transform(bat)
bat_pca = PCA(n_components=2)
bat_2d = bat_pca.fit_transform(bat_scaled)
print("Batter PCA explained variance:", bat_pca.explained_variance_ratio_.round(3))

plt.figure(figsize=(9, 7))
plt.scatter(bat_2d[:, 0], bat_2d[:, 1], c=bat["strike_rate"], cmap="viridis", s=80)
for name, (x, y) in zip(bat.index, bat_2d):
    plt.annotate(name, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)
plt.colorbar(label="Strike rate")
plt.xlabel("First principal component")
plt.ylabel("Second Principal Component")
plt.title("Batter-level PCA")
plt.tight_layout()
plt.savefig("pca_batter_scatter.png", dpi=150)
plt.show()

bat_comp = pd.DataFrame(bat_pca.components_, columns=bat.columns, index=["PC1", "PC2"])
plt.figure(figsize=(12, 3.5))
sns.heatmap(bat_comp, cmap="plasma", annot=True, fmt=".2f")
plt.title("Batter-level component loadings")
plt.tight_layout()
plt.savefig("pca_batter_heatmap.png", dpi=150)
plt.show()
