import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("C:/Users/homec/Downloads/TMDB_movie_dataset_v11.csv")


#Applying Filters to Features
df = df[~(
    (df['vote_average'] == 0) |
    (df['vote_count'] <= 50) |
    (df['runtime'] <= 60) |
    (df['budget'] <= 100_000) |
    (df['revenue'] <= 1_000_000))]

y = df['revenue']

#Pick Out Features
features = ['vote_average', 'vote_count', 'runtime', 'budget', 'popularity']
X = df[features].copy()

#Log Transforming Skewed Features
X['budget_log'] = np.log1p(X['budget'])
X['popularity_log'] = np.log1p(X['popularity'])
X['vote_count_log'] = np.log1p(X['vote_count'])

#Feature Engeneering
X['vote_weighted'] = X['vote_average'] * X['vote_count_log']
X['budget_per_runtime'] = X['budget_log'] * X['runtime']
X['popularity_per_vote'] = X['popularity_log'] * X['vote_count_log']

X = X.fillna(0)

#Split data with 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

mod = RandomForestRegressor(
    n_estimators= 500,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

#Train the model
mod.fit(X_train, y_train_log)

#Predict and inverse log-transform
y_pred = np.expm1(mod.predict(X_test))
y_pred_log = mod.predict(X_test)
y_test_actual = y_test

rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
r2 = r2_score(y_test_actual, y_pred)
print("RMSE:", rmse)
print("R²:", r2)

rmse_log = np.sqrt(mean_squared_error(y_test_log, y_pred_log))
r2_log = r2_score(y_test_log, y_pred_log)
print("Log-space RMSE:", rmse_log)
print("Log-space R²:", r2_log)

predictions_df = pd.DataFrame({
    'Actual Revenue (M$)': y_test_actual.values / 1e6,
    'Predicted Revenue (M$)': y_pred / 1e6
})

predictions_log_df = pd.DataFrame({
    'Actual Log-Revenue': y_test_log.values,
    'Predicted Log-Revenue': y_pred_log
})
print(predictions_log_df.head())
print(predictions_df.head())




