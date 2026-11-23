"""
K-Nearest Neighbors on ball-by-ball cricket data
Follows the structure of the KNN lab PDF:
imports -> data -> standardize -> train/test split -> KNN (k=1) -> evaluate
-> elbow method (error rate vs K) -> retrain with better K -> confusion matrix.

Task: predict whether a ball results in a WICKET (1) or not (0).
Keep synthetic_cricket_ball_by_ball_10k.csv in the same folder.
"""

# ---------------- Import Libraries ----------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_auc_score, recall_score)

# ---------------- Get the Data ----------------
df = pd.read_csv("synthetic_cricket_ball_by_ball_10k.csv")
print(df.shape)

# TARGET CLASS (the PDF's 'TARGET CLASS' column): 1 = wicket, 0 = no wicket
df["TARGET CLASS"] = df["wicket_type"].notna().astype(int)
print(df["TARGET CLASS"].value_counts())

# runs_off_bat, extras, total_runs, wicket_type, player_out are OUTCOMES of the
# ball -> excluded to avoid data leakage.
feature_cols = ["innings", "over", "ball", "batting_team", "bowling_team",
                "striker", "bowler", "venue", "pitch_type", "weather", "match_phase"]
X = pd.get_dummies(df[feature_cols]).astype(float)   # KNN needs numeric input
y = df["TARGET CLASS"]
print(X.shape)

# ---------------- Train Test Split ----------------
# The PDF standardizes BEFORE splitting. Better practice: split first, then fit the
# scaler on the training data only, so no test information leaks into the scaler.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=101, stratify=y)

# ---------------- Standardize the Variables ----------------
# KNN is distance based, so variables on a large scale dominate the distance.
scaler = StandardScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

df_feat = pd.DataFrame(X_train_s, columns=X.columns)
print(df_feat.head())

# ---------------- Using KNN (start with k=1) ----------------
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_s, y_train)
pred = knn.predict(X_test_s)

print("\n=== KNN, k=1 ===")
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, zero_division=0))

# ---------------- Choosing a K value (elbow method) ----------------
# The PDF plots the test error rate for k = 1..39 and picks the lowest.
# Here the classes are imbalanced (~4.7% wickets), so a big K just predicts
# "no wicket" for everything and the error rate looks great (~4.7%) while recall is 0.
# So we also track recall and ROC AUC for the wicket class.
k_range = range(1, 40)
error_rate, recall_rate, auc_rate = [], [], []
for i in k_range:
    knn_i = KNeighborsClassifier(n_neighbors=i).fit(X_train_s, y_train)
    pred_i = knn_i.predict(X_test_s)
    error_rate.append(np.mean(pred_i != y_test))
    recall_rate.append(recall_score(y_test, pred_i, zero_division=0))
    auc_rate.append(roc_auc_score(y_test, knn_i.predict_proba(X_test_s)[:, 1]))

plt.figure(figsize=(10, 6))
plt.plot(k_range, error_rate, color="blue", linestyle="dashed", marker="o",
         markerfacecolor="red", markersize=8)
plt.title("Error Rate vs. K Value")
plt.xlabel("K")
plt.ylabel("Error Rate")
plt.savefig("knn_error_vs_k.png", dpi=120)
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].plot(k_range, recall_rate, marker="o", color="green")
ax[0].set_title("Wicket recall vs. K"); ax[0].set_xlabel("K"); ax[0].set_ylabel("Recall")
ax[1].plot(k_range, auc_rate, marker="o", color="purple")
ax[1].set_title("ROC AUC vs. K"); ax[1].set_xlabel("K"); ax[1].set_ylabel("ROC AUC")
plt.tight_layout()
plt.savefig("knn_recall_auc_vs_k.png", dpi=120)
plt.show()

# Pick K by 5-fold cross-validated ROC AUC on the TRAINING set (not on the test set),
# so the final test score stays an honest estimate.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=101)
cv_auc = [cross_val_score(KNeighborsClassifier(n_neighbors=k), X_train_s, y_train,
                          cv=cv, scoring="roc_auc").mean() for k in range(5, 80, 2)]
best_k = list(range(5, 80, 2))[int(np.argmax(cv_auc))]
print("\nBest K by cross-validated ROC AUC:", best_k, "| CV AUC:", round(max(cv_auc), 3))

# ---------------- Retrain with the new K ----------------
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_s, y_train)
pred = knn.predict(X_test_s)
proba = knn.predict_proba(X_test_s)[:, 1]

print(f"\n=== KNN, k={best_k} (default 0.5 threshold) ===")
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, zero_division=0))
print("Test ROC AUC:", round(roc_auc_score(y_test, proba), 3))

# With 4.7% wickets, the 0.5 probability threshold almost never fires.
# Lowering the threshold trades precision for recall on the wicket class.
thr = np.quantile(proba, 0.80)       # flag the top 20% most "wicket-like" balls
pred_thr = (proba >= thr).astype(int)
print(f"\n=== KNN, k={best_k}, threshold lowered to {thr:.3f} ===")
print(confusion_matrix(y_test, pred_thr))
print(classification_report(y_test, pred_thr, zero_division=0))

# ---------------- Confusion matrix plot ----------------
conf_matrix = confusion_matrix(y_test, pred_thr)
vis = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                             display_labels=["No wicket", "Wicket"])
vis.plot()
plt.grid(False)
plt.title(f"KNN (k={best_k}, lowered threshold)")
plt.savefig("knn_confusion_matrix.png", dpi=120)
plt.show()
