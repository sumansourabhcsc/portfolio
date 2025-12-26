import streamlit as st
import pandas as pd
from datetime import datetime

from utils.auth import global_auth
from utils.data_loader import (
    load_fund_csv,
    fetch_nav_history,
    fetch_latest_nav,
    load_history_from_github,
)
from utils.calculations import (
    compute_fund_metrics,
    compute_portfolio_overview,
    compute_xirr,
)
from utils.github_sync import update_github_daily_summary
from utils.formatting import format_indian
from utils.ui import render_header, render_footer, render_nav_chart

from config.funds import mutual_funds, fund_logos, SCHEME_URLS


# -----------------------------------------------------------------------------
# PAGE CONFIG + AUTH
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Mutual Fund Dashboard", layout="wide")
global_auth()

render_header()

st.sidebar.header("Your Mutual Funds")
overview_button = st.sidebar.button("📦 Complete Overview")

st.sidebar.markdown("---")
selected_fund = st.sidebar.radio(
    "Single Fund View", list(mutual_funds.keys()), index=0
)


# -----------------------------------------------------------------------------
# COMPLETE PORTFOLIO OVERVIEW
# -----------------------------------------------------------------------------
if overview_button:
    st.title("📦 Complete Portfolio Overview")

    # Load all funds
    all_funds_data = []
    for fund_name, code in mutual_funds.items():
        df = load_fund_csv(fund_name)
        if df is not None:
            all_funds_data.append((fund_name, code, df))

    # Compute portfolio metrics
    portfolio_summary, per_fund_summary, latest_date = compute_portfolio_overview(
        all_funds_data
    )

    # Display summary
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Invested", f"₹ {portfolio_summary['invested']:,.2f}")
    col2.metric("Total Current", f"₹ {portfolio_summary['current']:,.2f}")
    col3.metric("Total Gain", f"₹ {portfolio_summary['gain']:,.2f}")
    col4.metric("Return %", f"{portfolio_summary['return_pct']:.2f}%")
    col5.metric("Portfolio XIRR", f"{portfolio_summary['xirr']:.2f}%")

    # Per‑fund table
    st.subheader("Individual Funds")
    st.dataframe(per_fund_summary, use_container_width=True)

    # Update GitHub daily summary
    update_github_daily_summary(
        portfolio_summary["invested"],
        portfolio_summary["current"],
        portfolio_summary["gain"],
        latest_date,
    )

    # Load daily history
    df_hist = load_history_from_github()
    if df_hist is not None:
        render_nav_chart(df_hist)

    render_footer()
    st.stop()


# -----------------------------------------------------------------------------
# SINGLE FUND VIEW
# -----------------------------------------------------------------------------
st.title(f"📄 {selected_fund}")

fund_code = mutual_funds[selected_fund]

# Load CSV
df = load_fund_csv(selected_fund)
if df is None:
    st.error("CSV not found for this fund.")
    st.stop()

# Fetch NAV history
start_date = df["Date"].min().date()
end_date = datetime.today().date()

nav_df = fetch_nav_history(fund_code, start_date, end_date)

# Compute metrics
metrics = compute_fund_metrics(df, nav_df)

# Display metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Invested", f"₹ {metrics['invested']:,.2f}")
col2.metric("Current Value", f"₹ {metrics['current']:,.2f}")
col3.metric("Gain", f"₹ {metrics['gain']:,.2f}")
col4.metric("Return %", f"{metrics['return_pct']:.2f}%")

# NAV chart
render_nav_chart(nav_df)

# Investment table
st.subheader("Investment Details")
st.dataframe(metrics["table"], use_container_width=True)

render_footer()
