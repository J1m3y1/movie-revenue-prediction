# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 22:06:37 2025

@author: homec
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import matplotlib.pyplot as plt

# -------------------------
# Load and prepare data
# -------------------------
df = pd.read_csv('TMDB_movie_dataset_v11.csv')
df = df[~((df['vote_average'] == 0) |
          (df['vote_count'] <= 50) |
          (df['runtime'] <= 60) |
          (df['budget'] <= 100_000) |
          (df['revenue'] <= 1_000_000))]

# Target variable in log-space
y = np.log1p(df['vote_average'])

# Base numeric features
features = ['revenue', 'vote_count', 'runtime', 'budget', 'popularity']
X = df[features].copy()

# -------------------------
# Feature Engineering
# -------------------------
X['budget_log'] = np.log1p(X['budget'])
X['popularity_log'] = np.log1p(X['popularity'])
X['vote_count_log'] = np.log1p(X['vote_count'])
X['budget_per_runtime_log'] = X['budget_log'] / X['runtime']
X['popularity_per_vote_log'] = X['popularity_log'] / X['vote_count_log']

X = X.fillna(0)

# -------------------------
# Train/Test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Pipeline with XGBRegressor
# -------------------------
pipe = Pipeline([
    ('clf', XGBRegressor(random_state=42, n_jobs=-1))
])

# -------------------------
# Bayesian hyperparameter search
# -------------------------
search_space = {
    'clf__learning_rate': Real(0.01, 0.1, prior='log-uniform'),
    'clf__n_estimators': Integer(100, 1000),
    'clf__max_depth': Integer(3, 10),
    'clf__subsample': Real(0.5, 1.0),
    'clf__colsample_bytree': Real(0.5, 1.0),
    'clf__gamma': Real(0, 5),
    'clf__min_child_weight': Integer(1, 10)
}

opt = BayesSearchCV(
    pipe,
    search_space,
    n_iter=30,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

opt.fit(X_train, y_train)

# -------------------------
# Results
# -------------------------
print("Best Parameters:", opt.best_params_)
print("Best CV R²:", opt.best_score_)

y_pred = opt.predict(X_test)
print("Test R²:", r2_score(y_test, y_pred))
print("Test MSE:", mean_squared_error(y_test, y_pred))

# -------------------------
# Feature importances
# -------------------------
xgb_model = opt.best_estimator_.named_steps['clf']
importances = xgb_model.feature_importances_
feature_names = X_train.columns

feat_imp = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

print("\nFeature Importances:")
print(feat_imp)

# Plot feature importances
plt.figure(figsize=(10,6))
plt.barh(feat_imp['feature'], feat_imp['importance'])
plt.gca().invert_yaxis()
plt.xlabel("Feature Importance")
plt.title("XGBoost Feature Importances")
plt.show()

# -------------------------
# Select top features
# -------------------------
top_features = feat_imp['feature'].head(5).tolist()
print("\nTop features selected for retraining:", top_features)

X_train_top = X_train[top_features]
X_test_top = X_test[top_features]

# Retrain XGB with top features only
xgb_top = XGBRegressor(random_state=42, n_jobs=-1)
xgb_top.fit(X_train_top, y_train)

y_pred_top = xgb_top.predict(X_test_top)
print("\nTest R² (top features):", r2_score(y_test, y_pred_top))
print("Test MSE (top features):", mean_squared_error(y_test, y_pred_top))

