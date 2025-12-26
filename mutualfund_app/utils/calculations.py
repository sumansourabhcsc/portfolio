import pandas as pd
import numpy as np
from datetime import datetime

# Newton-Raphson XIRR
def xirr(cashflows, dates, guess=0.1):
    days = [(d - dates[0]).days for d in dates]

    def npv(rate):
        return sum(cf / ((1 + rate) ** (d / 365)) for cf, d in zip(cashflows, days))

    def d_npv(rate):
        return sum(-(d/365) * cf / ((1 + rate) ** (1 + d/365)) for cf, d in zip(cashflows, days))

    rate = guess
    for _ in range(100):
        f = npv(rate)
        df = d_npv(rate)
        if df == 0:
            break
        new_rate = rate - f/df
        if abs(new_rate - rate) < 1e-6:
            return new_rate
        rate = new_rate
    return rate

def compute_portfolio_xirr(all_funds, total_current, latest_dates):
    cashflows = []
    dates = []

    for fund_name, df in all_funds:
        for _, row in df.iterrows():
            cashflows.append(-float(row["Amount"]))
            dates.append(pd.to_datetime(row["Date"]))

    final_date = max(latest_dates)
    cashflows.append(total_current)
    dates.append(final_date)

    combined = sorted(zip(dates, cashflows), key=lambda x: x[0])
    dates, cashflows = zip(*combined)

    return xirr(list(cashflows), list(dates))
