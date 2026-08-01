import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from forecast import prepare_daily_sales, FEATURE_COLUMNS

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, "data", "sales_data.csv")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "sales_forecast_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.txt")
os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("sales_data.csv not found. Run: python src/generate_data.py")

df = pd.read_csv(DATA_PATH)
daily = prepare_daily_sales(df)

X = daily[FEATURE_COLUMNS]
y = daily["revenue"]

split_index = int(len(daily) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=12)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

joblib.dump(model, MODEL_PATH)

with open(METRICS_PATH, "w") as f:
    f.write(f"MAE: {mae:.2f}\n")
    f.write(f"RMSE: {rmse:.2f}\n")
    f.write(f"MAPE: {mape:.2f}%\n")

print("Model trained successfully")
print(f"Saved model: {MODEL_PATH}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
