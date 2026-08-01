import os
import numpy as np
import pandas as pd
from datetime import datetime

np.random.seed(42)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

products = [
    ("Laptop", "Electronics", 55000),
    ("Smartphone", "Electronics", 25000),
    ("Headphones", "Electronics", 2500),
    ("Office Chair", "Furniture", 7500),
    ("Study Table", "Furniture", 12000),
    ("T-Shirt", "Fashion", 799),
    ("Jeans", "Fashion", 1799),
    ("Shoes", "Fashion", 2499),
    ("Coffee Maker", "Home Appliances", 4500),
    ("Mixer Grinder", "Home Appliances", 3500),
]

regions = ["South", "North", "East", "West"]
stores = ["Store A", "Store B", "Store C", "Store D", "Store E"]

dates = pd.date_range(start="2023-01-01", end="2025-12-31", freq="D")
rows = []

festival_months = [8, 10, 11, 12]

for date in dates:
    for product_name, category, base_price in products:
        region = np.random.choice(regions)
        store = np.random.choice(stores)
        weekend_boost = 1.25 if date.dayofweek >= 5 else 1.0
        festival_boost = 1.45 if date.month in festival_months else 1.0
        promotion = np.random.choice([0, 1], p=[0.72, 0.28])
        promo_boost = 1.35 if promotion else 1.0
        holiday = 1 if (date.month == 12 and date.day in [24, 25, 31]) or (date.month == 1 and date.day == 1) else 0
        holiday_boost = 1.5 if holiday else 1.0

        category_factor = {
            "Electronics": 1.1,
            "Fashion": 1.35,
            "Furniture": 0.75,
            "Home Appliances": 0.85,
        }[category]

        demand = 8 * weekend_boost * festival_boost * promo_boost * holiday_boost * category_factor
        quantity = max(1, int(np.random.poisson(demand)))
        discount = np.random.choice([0, 5, 10, 15, 20], p=[0.35, 0.25, 0.2, 0.12, 0.08])
        price = base_price * np.random.uniform(0.95, 1.08)
        revenue = quantity * price * (1 - discount / 100)
        cost = base_price * np.random.uniform(0.60, 0.78)
        profit = revenue - (quantity * cost)

        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "product_name": product_name,
            "category": category,
            "region": region,
            "store": store,
            "quantity": quantity,
            "price": round(price, 2),
            "discount": discount,
            "revenue": round(revenue, 2),
            "profit": round(profit, 2),
            "promotion": promotion,
            "holiday": holiday,
        })

sales_df = pd.DataFrame(rows)
output_path = os.path.join(DATA_DIR, "sales_data.csv")
sales_df.to_csv(output_path, index=False)
print(f"Dataset created: {output_path}")
print(sales_df.head())
