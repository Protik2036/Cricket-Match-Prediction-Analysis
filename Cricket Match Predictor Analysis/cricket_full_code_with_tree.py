"""
Decision Tree + Random Forest on ball-by-ball cricket data
Task: predict whether a ball results in a WICKET (1) or not (0).
Run:  python cricket_full_code_with_tree.py
Keep synthetic_cricket_ball_by_ball_10k.csv in the same folder.
"""

# ---------------- Import Libraries ----------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ---------------- Get the Data ----------------
df = pd.read_csv("synthetic_cricket_ball_by_ball_10k.csv")
print(df.shape)
print(df.columns)
print(df.isna().sum())

# Target: wicket fell on this ball (wicket_type is NaN when no wicket)
df["is_wicket"] = df["wicket_type"].notna().astype(int)
print(df["is_wicket"].value_counts(normalize=True))

# ---------------- EDA ----------------
sns.countplot(x="is_wicket", hue="is_wicket", data=df, palette="Set1", legend=False)
plt.title("Wicket (1) vs No wicket (0)")
plt.savefig("eda_target_balance.png", dpi=120)
plt.show()

sns.pairplot(df[["over", "ball", "runs_off_bat", "extras", "is_wicket"]]
             .sample(1000, random_state=42), hue="is_wicket", palette="Set1")
plt.savefig("eda_pairplot.png", dpi=100)
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, col in zip(axes.ravel(),
                   ["match_phase", "pitch_type", "weather", "venue", "innings", "bowler"]):
    df.groupby(col)["is_wicket"].mean().sort_values().plot.barh(ax=ax)
    ax.set_title(f"Wicket rate by {col}")
plt.tight_layout()
plt.savefig("eda_wicket_rates.png", dpi=100)
plt.show()

# ---------------- Features ----------------
# runs_off_bat, extras, total_runs, wicket_type, player_out are OUTCOMES of the
# ball, so they are excluded (data leakage).
feature_cols = ["innings", "over", "ball", "batting_team", "bowling_team",
                "striker", "bowler", "venue", "pitch_type", "weather", "match_phase"]
X = pd.get_dummies(df[feature_cols])
y = df["is_wicket"]

# ---------------- Train Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=101, stratify=y)

num_cols = ["innings", "over", "ball"]
scaler = StandardScaler()
X_train, X_test = X_train.copy(), X_test.copy()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# ---------------- Decision Tree (full, overfits) ----------------
dtree = DecisionTreeClassifier(criterion="entropy", class_weight="balanced",
                               random_state=101)
dtree.fit(X_train, y_train)
pred_full = dtree.predict(X_test)
print("\n=== Full Decision Tree ===")
print("Train acc:", dtree.score(X_train, y_train), "| Test acc:", dtree.score(X_test, y_test))
print(confusion_matrix(y_test, pred_full))
print(classification_report(y_test, pred_full, zero_division=0))

# ---------------- Decision Tree (pruned) ----------------
dtree_pruned = DecisionTreeClassifier(criterion="entropy", max_depth=4,
                                      min_samples_leaf=50, class_weight="balanced",
                                      random_state=101)
dtree_pruned.fit(X_train, y_train)
pred_pruned = dtree_pruned.predict(X_test)
print("\n=== Pruned Decision Tree (max_depth=4) ===")
print("Train acc:", dtree_pruned.score(X_train, y_train), "| Test acc:", dtree_pruned.score(X_test, y_test))
print(confusion_matrix(y_test, pred_pruned))
print(classification_report(y_test, pred_pruned, zero_division=0))

# ---------------- DISPLAY THE DECISION TREE ----------------
# 1) Graphical tree
plt.figure(figsize=(24, 10))
plot_tree(dtree_pruned,
          feature_names=X.columns,
          class_names=["no wicket", "wicket"],
          rounded=True,
          filled=True,          # colour by class
          proportion=True,
          fontsize=8)
plt.title("Pruned Decision Tree - Wicket Prediction")
plt.savefig("cricket_tree.png", dpi=150, bbox_inches="tight")
plt.show()                      # opens a window

# 2) Text version of the tree
print("\n=== Tree rules (text) ===")
print(export_text(dtree_pruned, feature_names=list(X.columns)))

# ---------------- Random Forest ----------------
rfc = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=20,
                             class_weight="balanced_subsample",
                             n_jobs=-1, random_state=101)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
print("\n=== Random Forest ===")
print("Train acc:", rfc.score(X_train, y_train), "| Test acc:", rfc.score(X_test, y_test))
print(confusion_matrix(y_test, rfc_pred))
print(classification_report(y_test, rfc_pred, zero_division=0))

imp = pd.Series(rfc.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
imp.sort_values().plot.barh(figsize=(8, 6), title="Random Forest - top 15 features")
plt.tight_layout()
plt.savefig("rf_feature_importance.png", dpi=120)
plt.show()

# ---------------- Comparison ----------------
print("\nROC AUC comparison")
for name, m in [("Full tree", dtree), ("Pruned tree", dtree_pruned), ("Random forest", rfc)]:
    print(f"{name:14s}: {roc_auc_score(y_test, m.predict_proba(X_test)[:, 1]):.3f}")
