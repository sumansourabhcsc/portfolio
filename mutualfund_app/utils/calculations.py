import pandas as pd
from datetime import datetime
from utils.xirr import xirr


def compute_fund_metrics(df, nav_df):
    latest_nav = nav_df["nav"].iloc[-1]

    df = df.copy()
    df["Current Value"] = df["Units"] * latest_nav
    df["Gain"] = df["Current Value"] - df["Amount"]

    invested = df["Amount"].sum()
    current = df["Current Value"].sum()
    gain = current - invested
    return_pct = (gain / invested * 100) if invested else 0

    return {
        "invested": invested,
        "current": current,
        "gain": gain,
        "return_pct": return_pct,
        "table": df,
    }


def compute_portfolio_overview(all_funds):
    total_invested = 0
    total_current = 0
    latest_dates = []
    per_fund_summary = []

    for fund_name, code, df in all_funds:
        nav, nav_date = fetch_latest_nav(code)
        latest_dates.append(pd.to_datetime(nav_date))

        units = df["Units"].sum()
        invested = df["Amount"].sum()
        current = units * nav
        gain = current - invested
        return_pct = (gain / invested * 100) if invested else 0

        per_fund_summary.append({
            "Fund": fund_name,
            "Invested": invested,
            "Current": current,
            "Gain": gain,
            "Return %": return_pct,
        })

        total_invested += invested
        total_current += current

    total_gain = total_current - total_invested
    total_return_pct = (total_gain / total_invested * 100)

    xirr_value = compute_xirr(all_funds, total_current, latest_dates)

    return (
        {
            "invested": total_invested,
            "current": total_current,
            "gain": total_gain,
            "return_pct": total_return_pct,
            "xirr": xirr_value,
        },
        pd.DataFrame(per_fund_summary),
        max(latest_dates),
    )
