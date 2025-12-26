
import pandas as pd
import requests
import base64
import io

def load_daily_summary():
    df = pd.read_csv("mutualfund_app/data/mutualfund/daily_summary.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")

def load_all_funds():
    df = pd.read_csv("mutualfund_app/data/mutualfund/all_funds_combined.csv")
    df["fundcode"] = df["fundcode"].astype(str)
    return df

def fetch_latest_nav(fund_code):
    url = f"https://api.mfapi.in/mf/{fund_code}/latest"
    try:
        r = requests.get(url, timeout=10).json()
        return float(r["data"][0]["nav"]), r["data"][0]["date"]
    except:
        return None, None
