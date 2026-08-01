import pandas as pd

FEATURE_COLUMNS = [
    "day_of_week",
    "month",
    "quarter",
    "is_weekend",
    "promotion_rate",
    "holiday_rate",
    "lag_1",
    "lag_7",
    "rolling_7",
    "rolling_30",
]


def prepare_daily_sales(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    daily = df.groupby("date").agg(
        revenue=("revenue", "sum"),
        quantity=("quantity", "sum"),
        profit=("profit", "sum"),
        promotion_rate=("promotion", "mean"),
        holiday_rate=("holiday", "mean"),
    ).reset_index()

    daily = daily.sort_values("date")
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month
    daily["quarter"] = daily["date"].dt.quarter
    daily["is_weekend"] = daily["day_of_week"].isin([5, 6]).astype(int)
    daily["lag_1"] = daily["revenue"].shift(1)
    daily["lag_7"] = daily["revenue"].shift(7)
    daily["rolling_7"] = daily["revenue"].shift(1).rolling(window=7).mean()
    daily["rolling_30"] = daily["revenue"].shift(1).rolling(window=30).mean()
    daily = daily.dropna().reset_index(drop=True)
    return daily


def make_future_forecast(model, daily_df: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    history = daily_df.copy().sort_values("date")
    forecasts = []

    for _ in range(days):
        next_date = history["date"].max() + pd.Timedelta(days=1)
        recent = history.sort_values("date")
        feature_row = {
            "day_of_week": next_date.dayofweek,
            "month": next_date.month,
            "quarter": next_date.quarter,
            "is_weekend": int(next_date.dayofweek in [5, 6]),
            "promotion_rate": recent["promotion_rate"].tail(30).mean(),
            "holiday_rate": 1 if (next_date.month == 12 and next_date.day in [24, 25, 31]) or (next_date.month == 1 and next_date.day == 1) else 0,
            "lag_1": recent["revenue"].iloc[-1],
            "lag_7": recent["revenue"].iloc[-7],
            "rolling_7": recent["revenue"].tail(7).mean(),
            "rolling_30": recent["revenue"].tail(30).mean(),
        }
        X_next = pd.DataFrame([feature_row])[FEATURE_COLUMNS]
        predicted_revenue = float(model.predict(X_next)[0])
        predicted_revenue = max(0, predicted_revenue)

        new_row = {
            "date": next_date,
            "revenue": predicted_revenue,
            "quantity": recent["quantity"].tail(30).mean(),
            "profit": recent["profit"].tail(30).mean(),
            **feature_row,
        }
        history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)
        forecasts.append({"date": next_date, "predicted_revenue": round(predicted_revenue, 2)})

    return pd.DataFrame(forecasts)
