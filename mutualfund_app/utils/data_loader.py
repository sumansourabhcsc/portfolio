import os
import io
import base64
import pandas as pd
import requests
from utils.formatting import detect_delimiter


BASE_FOLDER = "mutualfund"


def load_fund_csv(fund_name):
    file_path = os.path.join(BASE_FOLDER, fund_name, "fund.csv")
    if not os.path.exists(file_path):
        return None

    with open(file_path, "rb") as f:
        raw = f.read()

    delim = detect_delimiter(raw)
    df = pd.read_csv(io.BytesIO(raw), sep=delim)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.sort_values("Date")
    return df


def fetch_nav_history(code, start, end):
    url = f"https://api.mfapi.in/mf/{code}?startDate={start}&endDate={end}"
    r = requests.get(url).json()

    df = pd.DataFrame(r["data"])
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
    df["nav"] = pd.to_numeric(df["nav"])
    return df.sort_values("date")


def fetch_latest_nav(code):
    url = f"https://api.mfapi.in/mf/{code}/latest"
    r = requests.get(url).json()
    return float(r["data"][0]["nav"]), r["data"][0]["date"]


def load_history_from_github():
    token = st.secrets["GITHUB_TOKEN"]
    username = st.secrets["GITHUB_USERNAME"]
    repo = st.secrets["GITHUB_REPO"]
    file_path = st.secrets["GITHUB_FILE_PATH"]

    api_url = f"https://api.github.com/repos/{username}/{repo}/contents/{file_path}"

    r = requests.get(api_url, headers={"Authorization": f"token {token}"})
    if r.status_code != 200:
        return None

    content = r.json()
    file_data = base64.b64decode(content["content"]).decode("utf-8")

    df = pd.read_csv(io.StringIO(file_data))
    df["date"] = pd.to_datetime(df["date"])
    return df
