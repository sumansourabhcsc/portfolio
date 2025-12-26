import base64
import io
import pandas as pd
import requests
import streamlit as st


def update_github_daily_summary(invested, current, gain, date):
    token = st.secrets["GITHUB_TOKEN"]
    username = st.secrets["GITHUB_USERNAME"]
    repo = st.secrets["GITHUB_REPO"]
    file_path = st.secrets["GITHUB_FILE_PATH"]

    api_url = f"https://api.github.com/repos/{username}/{repo}/contents/{file_path}"

    today = date.strftime("%Y-%m-%d")

    r = requests.get(api_url, headers={"Authorization": f"token {token}"})

    if r.status_code == 200:
        content = r.json()
        sha = content["sha"]
        file_data = base64.b64decode(content["content"]).decode("utf-8")
        df = pd.read_csv(io.StringIO(file_data))

        if today in df["date"].values:
            df.loc[df["date"] == today, ["total_invested", "total_current", "total_gain"]] = [
                invested, current, gain
            ]
        else:
            df.loc[len(df)] = [today, invested, current, gain]

        updated_data = df.to_csv(index=False)
    else:
        sha = None
        updated_data = (
            "date,total_invested,total_current,total_gain\n"
            f"{today},{invested},{current},{gain}\n"
        )

    encoded = base64.b64encode(updated_data.encode()).decode()

    payload = {"message": f"Daily update {today}", "content": encoded, "sha": sha}

    requests.put(api_url, json=payload, headers={"Authorization": f"token {token}"})
