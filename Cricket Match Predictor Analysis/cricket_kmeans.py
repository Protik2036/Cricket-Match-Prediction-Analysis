"""
K-Means clustering on synthetic ball-by-ball cricket data
(follows the same workflow as the k-means-clustering lab PDF)
"""

# ---------------------------------------------------------------
# 1. Import libraries
# ---------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ---------------------------------------------------------------
# 2. Import dataset
# ---------------------------------------------------------------
df = pd.read_csv('synthetic_cricket_ball_by_ball_10k.csv')

# ---------------------------------------------------------------
# 3. Exploratory data analysis
# ---------------------------------------------------------------
print("Shape:", df.shape)
print(df.head())
df.info()
print("\nMissing values:\n", df.isnull().sum())
print(df.describe())

# wicket_type / player_out are NaN when no wicket fell (not "bad" data),
# so convert them into a useful 0/1 feature instead of dropping rows.
df['is_wicket'] = df['wicket_type'].notnull().astype(int)

# ---------------------------------------------------------------
# 4. Drop identifier-like / non-numeric columns
#    (same idea as dropping status_id & status_published in the PDF)
# ---------------------------------------------------------------
df = df.drop(columns=['match_id', 'batting_team', 'bowling_team', 'striker',
                      'bowler', 'wicket_type', 'player_out', 'venue',
                      'pitch_type', 'weather'])

# ---------------------------------------------------------------
# 5. Convert categorical variable into integers
#    match_phase has a natural order, so map it explicitly
#    (LabelEncoder would order it alphabetically: Death, Middle, Powerplay)
# ---------------------------------------------------------------
df['match_phase'] = df['match_phase'].map({'Powerplay': 0, 'Middle': 1, 'Death': 2})

print("\nFeatures used for clustering:\n", df.head())

# ---------------------------------------------------------------
# 6. Feature scaling (MinMaxScaler, as in the PDF)
# ---------------------------------------------------------------
cols = df.columns
X = MinMaxScaler().fit_transform(df)
X_df = pd.DataFrame(X, columns=cols)

# ---------------------------------------------------------------
# 7. Elbow method + silhouette score to choose K
# ---------------------------------------------------------------
clustering_score, sil_score = [], []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = km.fit_predict(X)
    clustering_score.append(km.inertia_)
    sil_score.append(silhouette_score(X, labels, sample_size=3000, random_state=42))

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].plot(K_range, clustering_score, marker='o')
ax[0].set(title='The Elbow Method', xlabel='No. of Clusters', ylabel='Clustering Score (inertia)')
ax[1].plot(K_range, sil_score, marker='o', color='green')
ax[1].set(title='Silhouette Score', xlabel='No. of Clusters', ylabel='Silhouette')
for a in ax: a.grid(alpha=.3)
plt.tight_layout()
plt.savefig('elbow_silhouette.png', dpi=150)
plt.show()

# ---------------------------------------------------------------
# 8. Fit the final K-Means model
#    Choose K from the elbow plot (change this value if you prefer another)
# ---------------------------------------------------------------
K = 4
kmeans = KMeans(n_clusters=K, init='k-means++', n_init=10, random_state=42)
pred = kmeans.fit_predict(X)
df['Cluster'] = pred

print("\nNumber of data points in each cluster:\n", df['Cluster'].value_counts().sort_index())

# ---------------------------------------------------------------
# 9. Cluster profile (average of each original feature per cluster)
# ---------------------------------------------------------------
print("\nCluster profile (mean values):\n", df.groupby('Cluster').mean().round(2))

# ---------------------------------------------------------------
# 10. Visualization
# ---------------------------------------------------------------
colors = ['brown', 'green', 'blue', 'purple', 'orange', 'teal', 'pink']

# 10a. Two raw features (like num_reactions vs num_comments in the PDF)
plt.figure(figsize=(10, 6))
for c in range(K):
    plt.scatter(X[pred == c, cols.get_loc('over')],
                X[pred == c, cols.get_loc('total_runs')],
                c=colors[c], label=f'Cluster {c}', alpha=.4, s=25)
plt.scatter(kmeans.cluster_centers_[:, cols.get_loc('over')],
            kmeans.cluster_centers_[:, cols.get_loc('total_runs')],
            s=250, c='red', marker='*', edgecolors='black', label='Centroid')
plt.xlabel('over (scaled)'); plt.ylabel('total_runs (scaled)')
plt.title('Cricket ball-by-ball clusters'); plt.legend(); plt.grid(alpha=.3)
plt.savefig('clusters_scatter.png', dpi=150)
plt.show()

# 10b. PCA 2-D view (all features at once)
pca = PCA(n_components=2, random_state=42)
Z = pca.fit_transform(X)
Zc = pca.transform(kmeans.cluster_centers_)
plt.figure(figsize=(10, 6))
for c in range(K):
    plt.scatter(Z[pred == c, 0], Z[pred == c, 1], c=colors[c], label=f'Cluster {c}', alpha=.4, s=25)
plt.scatter(Zc[:, 0], Zc[:, 1], s=250, c='red', marker='*', edgecolors='black', label='Centroid')
plt.xlabel('PC1'); plt.ylabel('PC2'); plt.title('Clusters in PCA space')
plt.legend(); plt.grid(alpha=.3)
plt.savefig('clusters_pca.png', dpi=150)
plt.show()

# 10c. Heatmap of centroids
plt.figure(figsize=(9, 4))
sns.heatmap(pd.DataFrame(kmeans.cluster_centers_, columns=cols), annot=True, fmt='.2f', cmap='viridis')
plt.title('Cluster centroids (scaled features)')
plt.ylabel('Cluster')
plt.tight_layout()
plt.savefig('centroid_heatmap.png', dpi=150)
plt.show()

# ---------------------------------------------------------------
# 11. Model parameters study
# ---------------------------------------------------------------
print("\nInertia:", kmeans.inertia_)
print("Silhouette score:", silhouette_score(X, pred, sample_size=3000, random_state=42))
print("Centroids (scaled):\n", pd.DataFrame(kmeans.cluster_centers_, columns=cols).round(3))
