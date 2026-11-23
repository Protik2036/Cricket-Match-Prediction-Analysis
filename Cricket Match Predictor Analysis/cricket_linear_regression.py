"""
Linear Regression on the cricket ball-by-ball dataset
(follows the steps of "Day 05: Linear Regression" lab: USA_Housing -> cricket)

Goal : predict the runs scored on a delivery (total_runs) from the match
       situation known BEFORE the ball is bowled.
Run  : python cricket_linear_regression.py [path/to/synthetic_cricket_ball_by_ball_10k.csv]
Needs: pandas numpy matplotlib seaborn scikit-learn
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

sns.set_style('whitegrid')
OUT = 'lr_outputs'; os.makedirs(OUT, exist_ok=True)

def save_show(name):
    plt.tight_layout(); plt.savefig(f'{OUT}/{name}.png', dpi=130); plt.show()

# ------------------------------------------------------------------ 1. Get the data
path = sys.argv[1] if len(sys.argv) > 1 else 'synthetic_cricket_ball_by_ball_10k.csv'
cricket = pd.read_csv(path)
print(cricket.head()); print(cricket.shape); cricket.info()
print(cricket.describe())
print('Columns:', list(cricket.columns))

# ------------------------------------------------------------------ 2. EDA
# distribution of the target (displot replaces the deprecated distplot used in the lab)
sns.displot(cricket['total_runs'], discrete=True, color='darkgreen')
plt.title('Distribution of total_runs per ball'); save_show('01_target_distribution')

# pairplot of numeric columns (sampled so it stays fast)
sns.pairplot(cricket[['over', 'ball', 'innings', 'total_runs']].sample(1500, random_state=101))
plt.savefig(f'{OUT}/02_pairplot.png', dpi=110); plt.show()

# average runs by categorical context
fig, ax = plt.subplots(1, 4, figsize=(20, 4))
for a, col in zip(ax, ['match_phase', 'pitch_type', 'weather', 'venue']):
    sns.barplot(x=col, y='total_runs', data=cricket, ax=a, palette='viridis', errorbar=None)
    a.set_title(f'Avg runs/ball by {col}'); a.tick_params(axis='x', rotation=30)
save_show('03_runs_by_category')

# ------------------------------------------------------------------ 3. Build X and y
# Address-like text columns can't be used directly -> one-hot encode categoricals.
# runs_off_bat / extras / wicket columns are NOT used: they are part of the outcome (leakage).
num_cols = ['innings', 'over', 'ball']
cat_cols = ['match_phase', 'pitch_type', 'weather', 'venue']
X = pd.get_dummies(cricket[num_cols + cat_cols], columns=cat_cols, drop_first=True).astype(float)
y = cricket['total_runs']

# correlation heatmap (like house.corr() in the lab)
house = pd.concat([X, y], axis=1)
plt.figure(figsize=(13, 10))
sns.heatmap(house.corr(), cmap='YlGnBu', annot=True, fmt='.2f', annot_kws={'size': 7})
plt.title('Correlation matrix (features + total_runs)'); save_show('04_correlation_heatmap')
print('\nCorrelation with total_runs:\n', house.corr()['total_runs'].drop('total_runs').sort_values(ascending=False).round(3))

# ------------------------------------------------------------------ 4. Train / test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
print('\ntrain:', X_train.shape, ' test:', X_test.shape)

# ------------------------------------------------------------------ 5. Create and train the model
lm = LinearRegression()
lm.fit(X_train, y_train)

# ------------------------------------------------------------------ 6. Model evaluation: coefficients
print('\nIntercept:', round(lm.intercept_, 4))
coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient']).sort_values('Coefficient')
print(coeff_df.round(4))

print('\nInterpreting the coefficients (holding all other features fixed):')
for feat, c in coeff_df['Coefficient'].items():
    direction = 'increase' if c > 0 else 'decrease'
    print(f'  a 1 unit increase in {feat:<28} is associated with a {direction} of {abs(c):.3f} runs per ball')

# standardised coefficients: comparable magnitudes (raw ones are not, since units differ)
sc = StandardScaler().fit(X_train)
lm_std = LinearRegression().fit(sc.transform(X_train), y_train)
std_coef = pd.Series(lm_std.coef_, X.columns).sort_values()
plt.figure(figsize=(8, 6))
std_coef.plot(kind='barh', color=np.where(std_coef > 0, 'seagreen', 'indianred'))
plt.axvline(0, color='k', lw=.8); plt.title('Standardised coefficients (effect of +1 std dev on runs/ball)')
save_show('05_standardised_coefficients')

# ------------------------------------------------------------------ 7. Predictions from our model
predictions = lm.predict(X_test)
print('\nFirst 5 predictions:', predictions[:5].round(3), '| actual:', y_test.values[:5])

plt.figure(figsize=(6, 6))
plt.scatter(y_test, predictions, alpha=.25, color='tab:blue')
plt.plot([0, 10], [0, 10], 'r--', label='perfect prediction')
plt.xlabel('Actual total_runs'); plt.ylabel('Predicted total_runs'); plt.title('Predicted vs actual'); plt.legend()
save_show('06_predicted_vs_actual')

# residual histogram
residuals = y_test - predictions
sns.displot(residuals, kde=True, bins=40, color='teal')
plt.title('Residual histogram'); plt.xlabel('actual - predicted'); save_show('07_residuals')

# residuals vs predictions (checks the "constant variance" assumption)
plt.figure(figsize=(7, 4)); plt.scatter(predictions, residuals, alpha=.25); plt.axhline(0, color='r', ls='--')
plt.xlabel('Predicted'); plt.ylabel('Residual'); plt.title('Residuals vs predicted'); save_show('08_residuals_vs_predicted')

# ------------------------------------------------------------------ 8. Regression evaluation metrics
print('\nMAE :', metrics.mean_absolute_error(y_test, predictions))
print('MSE :', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print('R^2 :', metrics.r2_score(y_test, predictions))

# compare with a "no-model" baseline that always predicts the training mean
base = DummyRegressor(strategy='mean').fit(X_train, y_train).predict(X_test)
print('\nBaseline (always predict the mean):')
print('MAE :', metrics.mean_absolute_error(y_test, base))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, base)))
print('R^2 :', metrics.r2_score(y_test, base))

# ------------------------------------------------------------------ 9. Use the model on a new delivery
def predict_runs(over, ball, innings, phase, pitch, weather, venue):
    """Expected runs for one delivery in the given situation."""
    row = pd.DataFrame(0.0, index=[0], columns=X.columns)
    row.loc[0, ['innings', 'over', 'ball']] = [innings, over, ball]
    for col in (f'match_phase_{phase}', f'pitch_type_{pitch}', f'weather_{weather}', f'venue_{venue}'):
        if col in row.columns: row.loc[0, col] = 1.0      # reference categories stay all-zero
    return float(lm.predict(row)[0])

print('\nExpected runs, over 19, ball 3, innings 2, Death, Batting Friendly, Sunny, Wankhede Stadium:',
      round(predict_runs(19, 3, 2, 'Death', 'Batting Friendly', 'Sunny', 'Wankhede Stadium'), 3))
print('Expected runs, over 10, ball 3, innings 1, Middle, Bowling Friendly, Sunny, Lords:',
      round(predict_runs(10, 3, 1, 'Middle', 'Bowling Friendly', 'Sunny', 'Lords'), 3))
