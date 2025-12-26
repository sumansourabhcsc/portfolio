import streamlit as st
import pandas as pd
from utils.data_loader import load_daily_summary

st.title("📊 Monthly Portfolio Analytics")

df = load_daily_summary()
df["month"] = df["date"].dt.to_period("M")

monthly = df.groupby("month").agg(
    total_invested=("total_invested", "last"),
    total_current=("total_current", "last"),
    total_gain=("total_gain", "last"),
)

monthly["mom_gain_change"] = monthly["total_gain"].diff()
monthly["mom_return_pct"] = monthly["total_current"].pct_change() * 100

vol_df = df.copy()
vol_df["daily_gain"] = vol_df["total_gain"].diff()
vol_df["month"] = vol_df["date"].dt.to_period("M")

volatility = vol_df.groupby("month").agg(
    daily_gain_std=("daily_gain", "std"),
    best_day_gain=("daily_gain", "max"),
    worst_day_gain=("daily_gain", "min"),
)

monthly = monthly.join(volatility)

latest = monthly.iloc[-1]

st.subheader(f"Summary for {latest.name}")
col1, col2, col3 = st.columns(3)
col1.metric("Total Invested", f"{latest.total_invested:,.0f}")
col2.metric("Total Current", f"{latest.total_current:,.0f}")
col3.metric("Total Gain", f"{latest.total_gain:,.0f}")

st.line_chart(monthly[["total_current", "total_invested"]])
st.line_chart(monthly[["total_gain"]])
st.line_chart(monthly[["mom_return_pct"]])

st.header("📉 Volatility")
st.bar_chart(monthly[["daily_gain_std"]])
st.bar_chart(monthly[["best_day_gain", "worst_day_gain"]])

st.header("📅 Month-over-Month Table")
st.dataframe(monthly.style.format("{:,.2f}"))
