import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import re

from sklearn.metrics import mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor



print("\n📥 Loading dataset...")

df = pd.read_csv("data/Foreign_Exchange_Rates.csv")

df.columns = df.columns.str.strip()

if 'Date' not in df.columns:
    df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])
df = df.sort_values('Date')

# convert to numeric
for col in df.columns:
    if col != 'Date':
        df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(axis=1, how='all')

print(f"✅ Dataset loaded with {len(df)} rows\n")


def create_features(df):
    df = df.copy()
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['dayofweek'] = df['Date'].dt.dayofweek
    return df


def create_lags(df, target, lags=[1, 2, 3, 7, 14, 30]):
    df = df.copy()
    for lag in lags:
        df[f'lag_{lag}'] = df[target].shift(lag)
    return df



os.makedirs("models", exist_ok=True)

exclude_cols = ['Date', 'year', 'month', 'day', 'dayofweek']
currency_cols = [col for col in df.columns if col not in exclude_cols]

print("💱 Currencies detected:")
for c in currency_cols:
    print(f"   - {c}")
print("\n" + "="*60)


results = []


for i, currency in enumerate(currency_cols, 1):

    print(f"\n[{i}/{len(currency_cols)}] 🚀 Training: {currency}")

    data = df[['Date', currency]].copy()

    if data[currency].isna().all():
        print("   ⚠️ Skipped (no data)")
        continue

    data = create_features(data)
    data = create_lags(data, currency)

    # force numeric
    for col in data.columns:
        if col != 'Date':
            data[col] = pd.to_numeric(data[col], errors='coerce')

    data = data.dropna()

    if len(data) < 100:
        print("   ⚠️ Skipped (not enough data)")
        continue

    train = data.iloc[:-60]
    test = data.iloc[-60:]

    X_train = train.drop(['Date', currency], axis=1)
    y_train = train[currency]

    X_test = test.drop(['Date', currency], axis=1)
    y_test = test[currency]

    # MODEL (silent mode)
    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        min_data_in_leaf=20,
        verbosity=-1
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    # metrics
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

    print(f"   📊 MAE:  {mae:.4f}")
    print(f"   📊 RMSE: {rmse:.4f}")
    print(f"   📊 MAPE: {mape:.2f}%")

    # save model (safe filename)
    safe_currency = re.sub(r'[^A-Za-z0-9_]', '_', currency)
    joblib.dump(model, f"models/{safe_currency}_model.pkl")

    results.append({
        "Currency": currency,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE (%)": round(mape, 2)
    })



print("\n" + "="*60)
print("📈 FINAL RESULTS\n")

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("MAPE (%)")

print(results_df.to_string(index=False))

print("\n✅ All models trained & saved successfully!\n")