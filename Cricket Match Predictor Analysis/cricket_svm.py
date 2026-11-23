"""
Support Vector Machines on ball-by-ball cricket data
Follows the structure of the SVM lab PDF (breast cancer example):
imports -> data -> DataFrame -> train/test split -> SVC -> evaluation
-> GridSearchCV (C, gamma, kernel) -> best model evaluation.

Task: predict whether a ball results in a WICKET (1) or not (0).
Keep synthetic_cricket_ball_by_ball_10k.csv in the same folder.
"""

# ---------------- Import Libraries ----------------
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, ConfusionMatrixDisplay)

# ---------------- Get the Data ----------------
df = pd.read_csv("synthetic_cricket_ball_by_ball_10k.csv")
print(df.shape)
df.info()

# Target (like cancer['target'] in the PDF): 1 = wicket, 0 = no wicket
df["Wicket"] = df["wicket_type"].notna().astype(int)
print(df["Wicket"].value_counts())

# ---------------- Set up DataFrame ----------------
# runs_off_bat, extras, total_runs, wicket_type, player_out are OUTCOMES of the
# ball -> excluded to avoid data leakage.
feature_cols = ["innings", "over", "ball", "batting_team", "bowling_team",
                "striker", "bowler", "venue", "pitch_type", "weather", "match_phase"]
df_feat = pd.get_dummies(df[feature_cols]).astype(float)   # SVM needs numeric input
df_target = pd.DataFrame(df["Wicket"], columns=["Wicket"])
print(df_feat.shape)
print(df_feat.head())

sns.countplot(x="Wicket", hue="Wicket", data=df_target, palette="Set1", legend=False)
plt.title("Wicket (1) vs No wicket (0)")
plt.show()

# ---------------- Train Test Split ----------------
X = df_feat
y = df["Wicket"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=101, stratify=y)

# The PDF says "it may also help to normalize the data" - SVMs are distance based,
# so scaling matters a lot (unlike trees). Fit the scaler on TRAIN only.
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ---------------- Train the Support Vector Classifier ----------------
model = SVC()                       # default: kernel='rbf', C=1, gamma='scale'
model.fit(X_train_s, y_train)
predictions = model.predict(X_test_s)

print("\n=== Default SVC ===")
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions, zero_division=0))
print("Number of support vectors per class:", model.n_support_)

# Same model but with class_weight='balanced' (wickets are only ~4.7% of balls,
# so the default model just predicts "no wicket" for almost everything)
model_bal = SVC(class_weight="balanced")
model_bal.fit(X_train_s, y_train)
pred_bal = model_bal.predict(X_test_s)
print("\n=== Default SVC with class_weight='balanced' ===")
print(confusion_matrix(y_test, pred_bal))
print(classification_report(y_test, pred_bal, zero_division=0))

# ---------------- Gridsearch ----------------
# Same idea as the PDF: try C, gamma and kernel. Two differences:
#  1) gamma has no effect on the linear kernel (the PDF's output shows identical
#     scores for every gamma with kernel='linear'), so we use two separate grids.
#  2) scoring='roc_auc' instead of accuracy, because accuracy is misleading when
#     95% of balls are "no wicket".
param_grid = [
    {"kernel": ["rbf"], "C": [0.1, 1, 10, 100], "gamma": [1, 0.1, 0.01, 0.001]},
    {"kernel": ["linear"], "C": [0.01, 0.1, 1]},
]

grid = GridSearchCV(SVC(class_weight="balanced"), param_grid,
                    scoring="roc_auc", refit=True, cv=3, verbose=2, n_jobs=-1)

start = time.time()
grid.fit(X_train_s, y_train)       # may take a while!
print(f"Grid search finished in {time.time() - start:.0f}s")

print("Best params :", grid.best_params_)
print("Best CV AUC :", round(grid.best_score_, 3))
print(grid.best_estimator_)

# ---------------- Evaluate best model ----------------
grid_predictions = grid.predict(X_test_s)
print("\n=== Tuned SVC ===")
print(confusion_matrix(y_test, grid_predictions))
print(classification_report(y_test, grid_predictions, zero_division=0))
print("Test ROC AUC:", round(roc_auc_score(y_test, grid.decision_function(X_test_s)), 3))

ConfusionMatrixDisplay.from_predictions(y_test, grid_predictions,
                                        display_labels=["no wicket", "wicket"], cmap="Blues")
plt.title("Tuned SVC - confusion matrix")
plt.savefig("svm_confusion_matrix.png", dpi=120)
plt.show()

# Top grid-search results
res = pd.DataFrame(grid.cv_results_)[["param_kernel", "param_C", "param_gamma",
                                      "mean_test_score"]]
print(res.sort_values("mean_test_score", ascending=False).head(10))

# ---------------- Visualise the SVM in 2D (like the PDF's margin figures) ----------------
# SVMs live in high dimensions here (72 features), so we project to 2 PCA components
# just to see the decision regions. This is for illustration only.
from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=101)
Z_train = pca.fit_transform(X_train_s)
svm2d = SVC(kernel="rbf", C=1, gamma="scale", class_weight="balanced").fit(Z_train, y_train)

xx, yy = np.meshgrid(np.linspace(Z_train[:, 0].min() - 1, Z_train[:, 0].max() + 1, 300),
                     np.linspace(Z_train[:, 1].min() - 1, Z_train[:, 1].max() + 1, 300))
zz = svm2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, zz, alpha=0.25, cmap="coolwarm")
plt.scatter(Z_train[:, 0], Z_train[:, 1], c=y_train, s=6, cmap="coolwarm")
plt.scatter(svm2d.support_vectors_[:, 0], svm2d.support_vectors_[:, 1],
            s=18, facecolors="none", edgecolors="k", linewidths=0.4, label="support vectors")
plt.title("SVM decision regions on 2 PCA components")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.legend()
plt.savefig("svm_pca_boundary.png", dpi=120)
plt.show()
