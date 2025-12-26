import requests
import base64
import pandas as pd
import io

def update_github_daily_summary(total_invested, total_current, total_gain, latest_date):
    token = st.secrets["GITHUB_TOKEN"]
    username = st.secrets["GITHUB_USERNAME"]
    repo = st.secrets["GITHUB_REPO"]
    file_path = st.secrets["GITHUB_FILE_PATH"]

    api_url = f"https://api.github.com/repos/{username}/{repo}/contents/{file_path}"

    today = latest_date.strftime("%Y-%m-%d")

    response = requests.get(api_url, headers={"Authorization": f"token {token}"})

    if response.status_code == 200:
        content = response.json()
        sha = content["sha"]
        file_data = base64.b64decode(content["content"]).decode("utf-8")
        df = pd.read_csv(io.StringIO(file_data))

        if today in df["date"].values:
            df.loc[df["date"] == today, ["total_invested", "total_current", "total_gain"]] = [
                total_invested, total_current, total_gain
            ]
        else:
            df.loc[len(df)] = [today, total_invested, total_current, total_gain]

        updated_data = df.to_csv(index=False)
    else:
        sha = None
        updated_data = f"date,total_invested,total_current,total_gain\n{today},{total_invested},{total_current},{total_gain}\n"

    encoded = base64.b64encode(updated_data.encode()).decode()

    payload = {"message": f"Daily update {today}", "content": encoded, "sha": sha}

    requests.put(api_url, json=payload, headers={"Authorization": f"token {token}"})
