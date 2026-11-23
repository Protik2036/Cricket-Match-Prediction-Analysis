# ============================================================
# Day-06 AI LAB: Logistic Regression on Cricket Ball-by-Ball Data
# Task: Predict whether a WICKET falls on a given ball (1 = Wicket, 0 = No wicket)
# Follows the same steps as the Titanic lab.
# ============================================================

# ---------- Import Libraries ----------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline   # uncomment if running in Jupyter

# ---------- The Data ----------
# Change the path if your file is in another folder
cricket = pd.read_csv('synthetic_cricket_ball_by_ball_10k.csv')

print(cricket.head())
print(cricket.tail())
print(cricket.shape)

# ============================================================
# Exploratory Data Analysis
# ============================================================

# ---------- Missing Data ----------
print(cricket.isnull().sum())

plt.figure(figsize=(10, 5))
sns.heatmap(cricket.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('Missing Data')
plt.show()

# wicket_type and player_out are only filled when a wicket falls,
# so the NaNs are meaningful, not "bad" data.
# We create our target variable from them:
cricket['Wicket'] = cricket['wicket_type'].notnull().astype(int)

# ---------- Target distribution ----------
sns.set_style('whitegrid')
sns.countplot(x='Wicket', data=cricket, palette='RdBu_r')
plt.title('Wicket (1) vs No Wicket (0)')
plt.show()
print(cricket['Wicket'].value_counts())
print(cricket['Wicket'].value_counts(normalize=True))

# ---------- Grouped count plots (hue=) ----------
plt.figure(figsize=(8, 5))
sns.countplot(x='Wicket', hue='match_phase', data=cricket, palette='rainbow')
plt.title('Wickets by Match Phase')
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x='Wicket', hue='pitch_type', data=cricket, palette='rocket_r')
plt.title('Wickets by Pitch Type')
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x='Wicket', hue='weather', data=cricket, palette='viridis')
plt.title('Wickets by Weather')
plt.show()

# Wicket rate (%) per category - easier to compare than raw counts
for col in ['match_phase', 'pitch_type', 'weather', 'venue', 'innings']:
    print(f"\nWicket rate by {col}:")
    print((cricket.groupby(col)['Wicket'].mean() * 100).round(2))

# ---------- Distributions of numeric columns ----------
sns.displot(cricket['over'], kde=False, color='darkblue', bins=20)
plt.title('Distribution of Over')
plt.show()

cricket['total_runs'].hist(bins=11, color='darkgreen', alpha=0.9)
plt.title('Distribution of Total Runs per Ball')
plt.show()

sns.countplot(x='extras', data=cricket, palette='rocket_r')
plt.title('Extras per Ball')
plt.show()

# Boxplot: over vs wicket
plt.figure(figsize=(8, 5))
sns.boxplot(x='Wicket', y='over', data=cricket, palette='winter')
plt.show()

# ============================================================
# Data Cleaning
# ============================================================
# Drop columns that should NOT be used as features:
#   - match_id        : just an identifier
#   - wicket_type     : only exists when a wicket falls (data leakage)
#   - player_out      : only exists when a wicket falls (data leakage)
#   - runs_off_bat, extras, total_runs : outcomes of the same ball
#                        (a wicket and runs are decided together, so we
#                         don't use them to "predict" the ball's result)
data = cricket.drop(['match_id', 'wicket_type', 'player_out',
                     'runs_off_bat', 'extras', 'total_runs'], axis=1)

print(data.info())

# ============================================================
# Converting Categorical Features (dummy variables)
# ============================================================
cat_cols = ['batting_team', 'bowling_team', 'striker', 'bowler',
            'venue', 'pitch_type', 'weather', 'match_phase']

data = pd.get_dummies(data, columns=cat_cols, drop_first=True)
print(data.head())
print(data.shape)

# ============================================================
# Building a Logistic Regression model
# ============================================================

# ---------- Train Test Split ----------
from sklearn.model_selection import train_test_split

X = data.drop('Wicket', axis=1)
y = data['Wicket']

# stratify keeps the same wicket ratio in train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=101, stratify=y)

# ---------- Training and Predicting ----------
from sklearn.linear_model import LogisticRegression

# Model 1: plain logistic regression (as in the lab)
logmodel = LogisticRegression(solver='lbfgs', max_iter=1000)
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)

# ---------- Evaluation ----------
from sklearn.metrics import (classification_report, accuracy_score,
                             precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_auc_score, roc_curve)

print("=== Model 1: Plain Logistic Regression ===")
print(classification_report(y_test, predictions, zero_division=0))

print("Accuracy :", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions, zero_division=0))
print("Recall   :", recall_score(y_test, predictions, zero_division=0))
print("F1 Score :", f1_score(y_test, predictions, zero_division=0))

conf_matrix = confusion_matrix(y_test, predictions)
print(conf_matrix)

# ---------- Wickets are rare (~5%), so accuracy is misleading ----------
# A model that ALWAYS predicts "no wicket" would score ~95% accuracy.
baseline_acc = 1 - y_test.mean()
print(f"\nBaseline accuracy (always predict 'No Wicket'): {baseline_acc:.4f}")

# Model 2: handle imbalance with class_weight='balanced'
logmodel_bal = LogisticRegression(solver='lbfgs', max_iter=1000,
                                  class_weight='balanced')
logmodel_bal.fit(X_train, y_train)
predictions_bal = logmodel_bal.predict(X_test)

print("\n=== Model 2: class_weight='balanced' ===")
print(classification_report(y_test, predictions_bal, zero_division=0))
print("Accuracy :", accuracy_score(y_test, predictions_bal))
print("Precision:", precision_score(y_test, predictions_bal, zero_division=0))
print("Recall   :", recall_score(y_test, predictions_bal, zero_division=0))
print("F1 Score :", f1_score(y_test, predictions_bal, zero_division=0))

conf_matrix_bal = confusion_matrix(y_test, predictions_bal)
print(conf_matrix_bal)

# ---------- Plotting a Confusion Matrix with Sklearn ----------
vis = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_bal,
                             display_labels=['No Wicket', 'Wicket'])
vis.plot()
plt.grid(False)
plt.title('Confusion Matrix (balanced model)')
plt.show()

# ---------- Confusion matrix with counts and percentages ----------
# Layout: [[TN, FP], [FN, TP]]  (rows = actual, columns = predicted)
group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
group_counts = ["{0:0.0f}".format(v) for v in conf_matrix_bal.flatten()]
group_percentages = ["{0:.2%}".format(v)
                     for v in conf_matrix_bal.flatten() / np.sum(conf_matrix_bal)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names, group_counts, group_percentages)]
labels = np.asarray(labels).reshape(2, 2)

sns.heatmap(conf_matrix_bal, annot=labels, fmt='', cmap='summer',
            xticklabels=['No Wicket', 'Wicket'],
            yticklabels=['No Wicket', 'Wicket'])
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()

# ---------- Sigmoid curve (from the lab theory) ----------
z = np.linspace(-10, 10, 200)
plt.plot(z, 1 / (1 + np.exp(-z)), color='red')
plt.axhline(0.5, color='black', linestyle='--', label='Threshold = 0.5')
plt.title('Sigmoid (Logistic) Function')
plt.xlabel('z')
plt.ylabel('Probability')
plt.legend()
plt.show()

# ---------- Predicted probabilities and custom threshold ----------
probs = logmodel_bal.predict_proba(X_test)[:, 1]
print("First 10 wicket probabilities:", np.round(probs[:10], 3))

# Try a different threshold than 0.5
threshold = 0.6
custom_pred = (probs >= threshold).astype(int)
print(f"\nThreshold = {threshold}")
print(classification_report(y_test, custom_pred, zero_division=0))

# ---------- ROC curve & AUC ----------
auc = roc_auc_score(y_test, probs)
fpr, tpr, _ = roc_curve(y_test, probs)
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# ---------- Most influential features ----------
coef = pd.Series(logmodel_bal.coef_[0], index=X.columns).sort_values()
plt.figure(figsize=(8, 6))
pd.concat([coef.head(8), coef.tail(8)]).plot(kind='barh', color='teal')
plt.title('Top Logistic Regression Coefficients')
plt.show()
