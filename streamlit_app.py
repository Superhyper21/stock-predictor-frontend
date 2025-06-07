import streamlit as st
import requests
import pandas as pd
import yfinance as yf
from datetime import timedelta

st.set_page_config(page_title="Stock Price Predictor", layout="centered")

st.title("📈 Stock Price Predictor with LSTM")

ticker = st.text_input("Enter stock ticker (e.g., AAPL, MSFT)", "AAPL")

if st.button("Predict"):
    try:
        with st.spinner("Predicting..."):
            response = requests.post("https://stock-predictor-backend.onrender.com/predict", json={"ticker": ticker})
            data = response.json()

        if "prediction" not in data:
            st.error(f"Error from server: {data.get('error', 'Unknown error')}")
        else:
            prediction = pd.Series(data["prediction"])
            hist = yf.download(ticker, period="90d")
            close_prices = hist['Close'][-60:]

            today = close_prices.index[-1]
            future_dates = pd.date_range(today + timedelta(days=1), periods=30, freq='B')

            real_df = pd.DataFrame({
                "Date": close_prices.index,
                "Price": close_prices.values,
                "Type": "Actual"
            })

            pred_df = pd.DataFrame({
                "Date": future_dates,
                "Price": prediction.values,
                "Type": "Predicted"
            })

            full_df = pd.concat([real_df, pred_df])
            st.subheader(f"📊 {ticker.upper()} - Actual vs Predicted Closing Prices")
            chart_data = full_df.set_index("Date")
            st.line_chart(chart_data, height=400, use_container_width=True)

            st.success("Prediction completed successfully!")

    except Exception as e:
        st.error(f"Failed to fetch prediction: {e}")
