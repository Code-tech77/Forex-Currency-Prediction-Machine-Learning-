import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re



st.set_page_config(
    page_title="Forex AI Predictor",
    page_icon="💱",
    layout="wide"
)

st.title("💱 Forex Currency Predictor")
st.markdown("Predict future exchange rates using LightGBM models")


@st.cache_data
def load_data():
    df = pd.read_csv("data/Foreign_Exchange_Rates.csv")

    df.columns = df.columns.str.strip()

    if 'Date' not in df.columns:
        df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values('Date')

    # convert numeric
    for col in df.columns:
        if col != 'Date':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


df = load_data()

exclude_cols = ['Date', 'year', 'month', 'day', 'dayofweek']
currencies = [col for col in df.columns if col not in exclude_cols]



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



def make_forecast(currency, days):

    # match model filename (sanitized)
    safe_currency = re.sub(r'[^A-Za-z0-9_]', '_', currency)
    model_path = f"models/{safe_currency}_model.pkl"

    if not os.path.exists(model_path):
        st.error("❌ Model not found! Train models first.")
        return None

    model = joblib.load(model_path)

    data = df[['Date', currency]].copy()

    predictions = []

    for _ in range(days):

        temp = create_features(data)
        temp = create_lags(temp, currency)

        # force numeric (critical)
        for col in temp.columns:
            if col != 'Date':
                temp[col] = pd.to_numeric(temp[col], errors='coerce')

        temp = temp.dropna()

        if temp.empty:
            return None

        last_row = temp.iloc[-1:]

        X = last_row.drop(['Date', currency], axis=1)

        pred = model.predict(X)[0]

        next_date = data['Date'].max() + pd.Timedelta(days=1)

        new_row = pd.DataFrame({
            'Date': [next_date],
            currency: [pred]
        })

        data = pd.concat([data, new_row], ignore_index=True)

        predictions.append((next_date, pred))

    return pd.DataFrame(predictions, columns=['Date', 'Prediction'])



col1, col2 = st.columns(2)

with col1:
    currency = st.selectbox("💱 Select Currency", currencies)

with col2:
    days = st.slider("📅 Forecast Days", 1, 60, 30)

if st.button("🚀 Generate Forecast"):

    with st.spinner("Generating predictions..."):
        forecast = make_forecast(currency, days)

    if forecast is not None:

        st.success("✅ Forecast generated successfully!")

        # metrics
        last_value = df[currency].dropna().iloc[-1]
        predicted_last = forecast['Prediction'].iloc[-1]

        change = ((predicted_last - last_value) / last_value) * 100

        m1, m2, m3 = st.columns(3)

        m1.metric("Current Value", f"{last_value:.4f}")
        m2.metric("Final Prediction", f"{predicted_last:.4f}")
        m3.metric("Change %", f"{change:.2f}%")

        # chart
        st.subheader("📈 Forecast Chart")
        st.line_chart(forecast.set_index('Date'))

        # table
        st.subheader("📊 Forecast Data")
        st.dataframe(forecast)

    else:
        st.error("⚠️ Could not generate forecast (data issue)")