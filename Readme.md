# 🪙 Bitcoin Price Prediction: Time-Series Analysis

This project is a machine learning implementation designed to predict Bitcoin's closing price using historical data. Built during my **30 Days of Project Building** challenge, it focuses on handling large datasets (100MB+) and time-series forecasting.

## 📌 Project Overview
The goal is to predict the next day's Bitcoin price based on the previous three days' performance. This project demonstrates data engineering skills like resampling 4 million rows of minute-by-minute data into a clean daily format.

## 🛠️ Tech Stack
- **Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn (Linear Regression)
- **Visualization:** Matplotlib



## 🚀 Key Features
* **Memory Optimization:** Uses specific data types to handle large CSV files efficiently in local environments like VS Code.
* **Data Resampling:** Transforms high-frequency (1-min) data into daily trends to reduce noise and computational load.
* **Lag Engineering:** Implements feature engineering by creating "lags" to capture temporal dependencies.
* **Evaluation:** Uses R-squared ($R^2$) metrics to validate prediction accuracy.

## 📂 Dataset
The model uses the **Bitcoin Historical Data** from Kaggle.
- **Source:** [Bitcoin Historical Data](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)
- **Timeframe:** 2012 to 2021
- **Size:** ~120 MB (Unzipped)

## 🔧 How to Use
1. **Install Libraries:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib
   ```

2. Run the Code:
    ```bash
    python main.py
    ```

