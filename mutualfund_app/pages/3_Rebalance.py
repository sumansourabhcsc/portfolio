import streamlit as st
import pandas as pd
from utils.data_loader import load_all_funds, fetch_latest_nav

st.title("🎯 Portfolio Rebalance Dashboard")

df = load_all_funds()

# Fetch NAVs
unique_codes = df["fundcode"].unique()
nav_records = []

for code in unique_codes:
    nav, nav_date = fetch_latest_nav(code)
    nav_records.append({"fundcode": code, "Latest NAV": nav, "Latest NAV Date": nav_date})

nav_df = pd.DataFrame(nav_records)
df = df.merge(nav_df, on="fundcode", how="left")

df["EffectiveNAV"] = df["Latest NAV"].fillna(df["NAV"])
df["CurrentValue"] = df["Units"] * df["EffectiveNAV"]

st.header("📁 Category Allocation")
category_summary = df.groupby("Category")["CurrentValue"].sum().reset_index()
category_summary["Weightage (%)"] = (
    category_summary["CurrentValue"] / category_summary["CurrentValue"].sum() * 100
).round(2)

st.dataframe(category_summary)
st.bar_chart(category_summary.set_index("Category")["Weightage (%)"])
