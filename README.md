# SalesVision - Sales Forecasting Data Science Project

SalesVision is an end-to-end data science project that analyzes historical sales data and forecasts future sales using machine learning.

## Features

- Synthetic retail sales dataset included
- Data cleaning and preprocessing
- Feature engineering for time series forecasting
- Model training using Random Forest Regression
- Forecast next 7, 30, 60, or 90 days
- Sales dashboard using Streamlit
- Product, category, and region analysis
- Model evaluation using MAE, RMSE, and MAPE

## Project Structure

```text
salesvision/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── sales_data.csv
├── models/
│   └── sales_forecast_model.pkl
└── src/
    ├── generate_data.py
    ├── train_model.py
    └── forecast.py
```

## How to Run

### 1. Open terminal inside project folder

```bash
cd salesvision
```

### 2. Create virtual environment

```bash
python -m venv venv
```

### 3. Activate virtual environment

Windows:

```bash
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Generate dataset

```bash
python src/generate_data.py
```

### 6. Train model

```bash
python src/train_model.py
```

### 7. Run dashboard

```bash
streamlit run app.py
```

## Dataset Columns

- date
- product_name
- category
- region
- store
- quantity
- price
- discount
- revenue
- profit
- promotion
- holiday

## Machine Learning Approach

The model predicts daily revenue using features such as:

- day of week
- month
- quarter
- weekend flag
- holiday flag
- promotion flag
- lag revenue
- 7-day rolling average
- 30-day rolling average

## Resume Description

Built an end-to-end Sales Forecasting and Business Intelligence dashboard using Python, Pandas, scikit-learn, and Streamlit. Implemented data preprocessing, feature engineering, Random Forest forecasting, business analytics, and interactive visualizations to predict future revenue and identify sales trends.

## Interview Explanation

This project solves a real business problem: predicting future sales from historical data. It helps businesses plan inventory, marketing campaigns, staffing, and revenue targets. The system first cleans and aggregates sales records, creates time-based features, trains a machine learning model, evaluates accuracy, and then displays forecasts through a dashboard.
