import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def find_csv_file(preferred=None):
    if preferred and os.path.exists(preferred):
        return preferred
    # look for any .csv in cwd
    files = [f for f in os.listdir('.') if f.lower().endswith('.csv')]
    if not files:
        raise FileNotFoundError('No CSV files found in the current directory.')
    # prefer exact-known common names
    for name in files:
        if 'bitstamp' in name.lower() or 'btcusd' in name.lower():
            return name
    return files[0]


def detect_columns(file_path):
    sample = pd.read_csv(file_path, nrows=5)
    cols = list(sample.columns)
    lc = [c.lower() for c in cols]

    # timestamp candidates
    ts_candidates = [c for c in cols if any(k in c.lower() for k in ('time', 'date', 'timestamp', 'unix'))]
    # close/price candidates
    price_candidates = [c for c in cols if any(k in c.lower() for k in ('close', 'price', 'last'))]

    if not ts_candidates or not price_candidates:
        raise ValueError(f"Couldn't auto-detect timestamp/price columns. Available columns: {cols}")

    ts_col = ts_candidates[0]
    price_col = price_candidates[0]

    # check sample to see if timestamp is epoch int
    sample_ts = sample[ts_col].iloc[0]
    is_epoch = False
    try:
        # numeric epoch values (large ints)
        if isinstance(sample_ts, (int, float)) or (isinstance(sample_ts, str) and sample_ts.isdigit()):
            is_epoch = True
    except Exception:
        is_epoch = False

    return ts_col, price_col, is_epoch


def main():
    # 1. LOAD DATA
    file_path = find_csv_file(preferred='bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv')

    try:
        ts_col, price_col, is_epoch = detect_columns(file_path)
    except Exception as e:
        print('Error detecting columns:', e)
        raise

    # read only necessary columns to save RAM
    df = pd.read_csv(file_path, usecols=[ts_col, price_col])
    df.columns = ['Timestamp', 'Close']

    # 2. CLEANING
    if is_epoch:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    else:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    df.dropna(inplace=True)

    # resample to daily mean to reduce size
    df.set_index('Timestamp', inplace=True)
    df = df.resample('D').mean()
    df.dropna(inplace=True)

    # 3. FEATURE ENGINEERING (Lag Features)
    for i in range(1, 4):
        df[f'Lag_{i}'] = df['Close'].shift(i)

    df.dropna(inplace=True)

    # 4. SPLITTING
    X = df[['Lag_1', 'Lag_2', 'Lag_3']].values
    y = df['Close'].values

    if len(y) <= 200:
        raise ValueError('Not enough data after resampling/lags to train and test (need >200 days).')

    X_train, X_test = X[:-100], X[-100:]
    y_train, y_test = y[:-100], y[-100:]

    # 5. MODELING
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 6. PREDICTION & VISUALIZATION
    predictions = model.predict(X_test)

    plt.figure(figsize=(12, 6))
    plt.plot(df.index[-100:], y_test, label='Actual Price', color='blue')
    plt.plot(df.index[-100:], predictions, label='Predicted Price', color='red', linestyle='--')
    plt.title('Bitcoin Price Prediction (Last 100 Days)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()

    print(f"Model Accuracy (R^2): {model.score(X_test, y_test):.4f}")


if __name__ == '__main__':
    main() 